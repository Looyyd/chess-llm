#!/usr/bin/env python
# coding: utf-8

import torch
import random
import chess
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from accelerate import PartialState
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import logging
from utils.chess_utils import (
    board_to_grid,
    extract_move_from_completion,
    parse_time_control,
)
from utils.dataset_utils import (
    select_weighted_position,
    reconstruct_board_position,
    get_turn_and_elo,
)

# In case previous experiments didn't close properly
torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG = False


# Add this custom callback class
class PeriodicInferenceCallback(TrainerCallback):
    """Callback to run inference periodically during training"""

    def __init__(self, tokenizer, test_prompt, inference_steps=500):
        self.tokenizer = tokenizer
        self.test_prompt = test_prompt
        self.inference_steps = inference_steps

    def on_step_end(self, args, state, control, **kwargs):
        # Run inference every `inference_steps` steps
        if state.global_step % self.inference_steps == 0 and state.global_step > 0:
            model = kwargs["model"]

            logger.info(f"\n{'='*50}")
            logger.info(f"Running inference at step {state.global_step}")
            logger.info(f"{'='*50}")

            # Prepare the input
            inputs = self.tokenizer(
                self.test_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(model.device)

            # Generate
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode and print
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Model response:\n{response}")

            # Extract the move from \boxed{} tags if present
            predicted_move = extract_move_from_completion(response)
            if predicted_move:
                logger.info(f"Predicted move: {predicted_move}")

            logger.info(f"{'='*50}\n")

            model.train()


def preprocess_chess_games(examples, tokenizer, min_elo=1500, min_time_control=300):
    """Preprocess a batch of chess games into training examples"""
    texts = []

    for i in range(len(examples["moves"])):
        moves = examples["moves"][i]
        white_elo = examples["white_elo"][i]
        black_elo = examples["black_elo"][i]
        time_control = examples["time_control"][i]
        # Opening data is useful to include, if at some point we add other data to SFT on, it might use the opening terminology
        # Also might help understand openings better?
        opening = examples.get("opening", ["Unknown"] * len(examples["moves"]))[i]
        eco = examples.get("eco", [""] * len(examples["moves"]))[i]

        # Skip if game is too short
        if len(moves) < 8:
            texts.append("")
            continue

        # Filter by ELO
        if white_elo < min_elo or black_elo < min_elo:
            texts.append("")
            continue

        base_time = parse_time_control(time_control)

        # Skip games with no time control or very fast games
        if base_time is None or base_time < 300:  # Less than 5 minutes
            texts.append("")
            continue

        # Select weighted position
        position_idx = select_weighted_position(moves)

        # Reconstruct board position
        board, move_history, move_history_str = reconstruct_board_position(
            moves, position_idx
        )

        # Get the next move (the answer)
        next_move = moves[position_idx]

        # Determine whose turn it is and their Elo
        turn, current_elo = get_turn_and_elo(board, white_elo, black_elo)

        # Create board visualization
        board_grid = board_to_grid(board)

        # Create the conversation
        system_prompt = "You are a chess engine. Given a chess position, predict the most likely next move based on the player's Elo rating and game context."

        user_prompt = f"""Current game position:

Player Elo: {current_elo}
Time Control: {time_control}
Opening: {opening}
ECO: {eco if eco else "N/A"}
Move history (UCI format): {move_history_str}
Turn: {turn}

Current board state:
{board_grid}


What is the most likely next move? Answer with the final answer only, inside an \\boxed{"{}"} box."""

        assistant_response = f"\\boxed{{{next_move}}}"

        # Format as messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)

    return {"text": texts}


def main():
    # Model configuration
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_size="left")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # device_map={"": device_string},
        attn_implementation="flash_attention_2",
    )

    # Load dataset from Hugging Face
    logger.info("Loading dataset from Hugging Face...")
    take_count = 1000 if DEBUG else None

    # Load from Hugging Face Hub
    dataset = load_dataset(
        "Looyyd/chess-dataset",
        data_files={"train": "train.jsonl"},
        split="train",
        streaming=True,
    )

    if take_count is not None:
        dataset = dataset.take(take_count)

    # Preprocess the dataset
    logger.info("Preprocessing dataset...")

    # Map preprocessing function with batching for efficiency
    train_dataset = dataset.map(
        lambda examples: preprocess_chess_games(examples, tokenizer),
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,  # Remove original columns, keep only 'text'
    )

    # Filter out empty texts (from games that were too short)
    train_dataset = train_dataset.filter(lambda example: len(example["text"]) > 0)

    # eval_dataset = dataset.skip(1000).take(200).map(...)

    # Training arguments
    training_args = SFTConfig(
        output_dir="./chess_sft_qwen",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        max_steps=1000 if DEBUG else 100_000,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=100,
        padding_free=True,  # this works with flash attention 2 and avoids padding errors, TODO: can now remove padding_left?
        # Disable local saving
        save_steps=500,
        save_total_limit=1,
        save_strategy="steps",
        eval_strategy="no",
        bf16=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        # SFT specific parameters
        max_length=1024,
        packing=False,  # Could enable for efficiency
        dataset_text_field="text",
        # Keep remove_unused_columns as default (True) since we already handled it in preprocessing
        report_to="none" if DEBUG else "wandb",
        push_to_hub=not DEBUG,
        hub_strategy="checkpoint",  # Saves at every save  with la latest checkpoint is also pushed in a subfolder allowing to resume training easily
        accelerator_config={
            # Otherwise the variable length sequences can cause issues on multi gpu
            "dispatch_batches": False,
        },
        use_liger_kernel=True,
    )

    # Prepare the test prompt
    test_position = """Current game position:

Player Elo: 1800
Time Control: 300+0
Opening: Ruy Lopez: Morphy Defense, Caro Variation
ECO: C70
Move history (UCI format): e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3 c6a5 b3c2 c7c5 d2d4 d8c7 b1d2 c5d4 c3d4 a5c6
Turn: White

Current board state:
  a b c d e f g h
  ----------------
8| r . b . . r k . |8
7| . . q . b p p p |7
6| p . n p . n . . |6
5| . p . . p . . . |5
4| . . . P P . . . |4
3| . . . . . N . P |3
2| P P B N . P P . |2
1| R . B Q R . K . |1
  ----------------
  a b c d e f g h


What is the most likely next move? Answer with the final answer only, inside an \\boxed{} box."""

    messages = [
        {
            "role": "system",
            "content": "You are a chess engine. Given a chess position, predict the most likely next move based on the player's Elo rating and game context.",
        },
        {"role": "user", "content": test_position},
    ]

    # Format with chat template
    test_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, padding_side="left"
    )

    # Create the inference callback
    inference_callback = PeriodicInferenceCallback(
        tokenizer=tokenizer,
        test_prompt=test_prompt,
        inference_steps=200,  # Run inference every 200 steps
    )

    # Initialize trainer with the callback
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,  # Uncomment if you have eval data
        processing_class=tokenizer,
        callbacks=[inference_callback],  # Add the callback here
    )

    # Fine-tune the model
    logger.info("Starting training...")
    trainer.train()

    # Save the model
    trainer.save_model()
    logger.info("Training complete! Model saved.")

    # Final test inference
    logger.info("\nFinal inference test:")
    inputs = tokenizer(
        test_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding_side="left",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Model response:\n{response}")

    # Extract the move from \boxed{} tags if present
    predicted_move = extract_move_from_completion(response)
    if predicted_move:
        print(f"\nPredicted move: {predicted_move}")


if __name__ == "__main__":
    main()
