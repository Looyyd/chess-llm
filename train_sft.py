#!/usr/bin/env python
# coding: utf-8

import torch
import random
import chess
import re
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,  # Add this import
)
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import logging

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
                self.test_prompt, return_tensors="pt", truncation=True, max_length=1024
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
            match = re.search(r"\\boxed\{([^}]+)\}", response)
            if match:
                predicted_move = match.group(1)
                logger.info(f"Predicted move: {predicted_move}")

            logger.info(f"{'='*50}\n")

            model.train()


def board_to_grid(board):
    """Convert board to visual grid representation"""
    grid_lines = []
    grid_lines.append("  a b c d e f g h")
    grid_lines.append("  ----------------")

    for rank in range(7, -1, -1):  # 8 to 1
        line = f"{rank + 1}|"
        for file in range(8):  # a to h
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece is None:
                line += " ."
            else:
                symbol = piece.symbol()
                # Use uppercase for white, lowercase for black
                line += f" {symbol}"
        line += f" |{rank + 1}"
        grid_lines.append(line)

    grid_lines.append("  ----------------")
    grid_lines.append("  a b c d e f g h")

    return "\n".join(grid_lines)


def preprocess_chess_games(examples, tokenizer):
    """Preprocess a batch of chess games into training examples"""
    texts = []

    for i in range(len(examples["moves"])):
        moves = examples["moves"][i]
        white_elo = examples["white_elo"][i]
        black_elo = examples["black_elo"][i]
        time_control = examples["time_control"][i]

        # Skip if game is too short
        if len(moves) < 2:
            # Add empty text to maintain batch size
            texts.append("")
            continue

        # Create weights that linearly increase from 1 to 5
        # position_idx can be from 0 to len(moves)-2 (inclusive)
        num_positions = len(moves) - 1

        # Create weights: weight = 1 + 4 * (position / (num_positions - 1))
        # This gives us 1 for position 0 and 5 for the last valid position
        weights = []
        for pos in range(num_positions):
            if num_positions == 1:
                weight = 1.0  # Only one position available
            else:
                weight = 1.0 + 4.0 * (pos / (num_positions - 1))
            weights.append(weight)

        # Use random.choices with weights to select position
        position_idx = random.choices(range(num_positions), weights=weights, k=1)[0]

        # Reconstruct board up to that position
        board = chess.Board()
        move_history = []

        for j in range(position_idx):
            move = moves[j]
            board.push_uci(move)
            move_history.append(move)

        # Get the next move (the answer)
        next_move = moves[position_idx]

        # Determine whose turn it is and their Elo
        if board.turn == chess.WHITE:
            current_elo = white_elo
            turn = "White"
        else:
            current_elo = black_elo
            turn = "Black"

        # Format move history
        move_history_str = " ".join(move_history) if move_history else "Game start"

        # Create board visualization
        board_grid = board_to_grid(board)

        # Create the conversation
        system_prompt = "You are a chess engine. Given a chess position, predict the most likely next move based on the player's Elo rating and game context."

        user_prompt = f"""Current game position:

Player Elo: {current_elo}
Time Control: {time_control}
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
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": device_string}
    )

    # Load dataset using load_dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files="./data/lichess_2013_12_compact.jsonl",
        split="train",
        streaming=True,  # Use streaming for large datasets
    )

    if DEBUG:
        # Take only first 1000 games for debugging
        dataset = dataset.take(1000)

    # Preprocess the dataset
    logger.info("Preprocessing dataset...")

    # Map preprocessing function with batching for efficiency
    train_dataset = dataset.map(
        lambda examples: preprocess_chess_games(examples, tokenizer),
        batched=True,
        batch_size=100,  # Process 100 games at a time
        remove_columns=dataset.column_names,  # Remove original columns, keep only 'text'
    )

    # Filter out empty texts (from games that were too short)
    train_dataset = train_dataset.filter(lambda example: len(example["text"]) > 0)

    # TODO: Create eval dataset by splitting the data
    # eval_dataset = dataset.skip(1000).take(200).map(...)

    # Training arguments
    training_args = SFTConfig(
        output_dir="./chess_lora_qwen",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        max_steps=1000 if DEBUG else 100_000,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        save_strategy="steps",
        eval_strategy="no",  # Set to "steps" if you have eval dataset
        bf16=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        # SFT specific parameters
        max_length=1024,
        packing=False,  # Could enable for efficiency
        dataset_text_field="text",
        # Keep remove_unused_columns as default (True) since we already handled it in preprocessing
        report_to="none" if DEBUG else "wandb",
        push_to_hub=not DEBUG,
        accelerator_config={
            # Otherwise the variable length sequences can cause issues on multi gpu
            "dispatch_batches": False,
        },
    )

    # Prepare the test prompt
    test_position = """Current game position:

Player Elo: 1800
Time Control: 180+0
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
        messages, tokenize=False, add_generation_prompt=True
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
        # Don't pass peft_config here since model is already wrapped
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
        test_prompt, return_tensors="pt", truncation=True, max_length=1024
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
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    if match:
        predicted_move = match.group(1)
        print(f"\nPredicted move: {predicted_move}")


if __name__ == "__main__":
    main()
