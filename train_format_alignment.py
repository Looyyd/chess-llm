#!/usr/bin/env python
# coding: utf-8

import torch
import os
import random
import chess
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from accelerate import PartialState
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import logging
from utils.chess_utils import (
    extract_move_from_completion,
    has_thinking_tags,
    board_to_grid,
)
from utils.dataset_utils import select_weighted_position, reconstruct_board_position
from utils.prompt_utils import (
    get_chess_system_prompt,
    create_chess_user_prompt,
    format_chess_messages,
)
import numpy as np

# In case previous experiments didn't close properly
torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG = False


class SimpleProgressCallback(TrainerCallback):
    """Simple callback that generates one response every 10 steps to show training progress"""

    def __init__(self, tokenizer, chess_dataset, inference_steps=10):
        self.tokenizer = tokenizer
        self.chess_dataset = chess_dataset
        self.inference_steps = inference_steps
        self.system_prompt = get_chess_system_prompt()

    def on_step_end(self, args, state, control, **kwargs):
        # Run inference every `inference_steps` steps
        if state.global_step % self.inference_steps == 0 and state.global_step > 0:
            model = kwargs["model"]

            logger.info(f"\n{'='*60}")
            logger.info(f"PROGRESS CHECK - Step {state.global_step}")
            logger.info(f"{'='*60}")

            # Create a random chess position prompt
            game_idx = random.randint(0, len(self.chess_dataset) - 1)
            game = self.chess_dataset[game_idx]
            moves = game["moves"]

            # Skip if game is too short
            if len(moves) < 8:
                return

            # Select weighted position
            position_idx = select_weighted_position(moves)

            # Reconstruct board position
            board, move_history, move_history_str = reconstruct_board_position(
                moves, position_idx
            )

            # Determine whose turn it is
            turn = "White" if board.turn == chess.WHITE else "Black"

            # Create board visualization
            board_grid = board_to_grid(board)

            # Create user prompt using shared utility
            user_prompt = create_chess_user_prompt(board_grid, move_history_str, turn)

            # Format as messages for chat template
            messages = format_chess_messages(self.system_prompt, user_prompt)

            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            logger.info(f"Position: {turn} to move after {position_idx} moves")
            logger.info(f"Board:\n{board_grid}")

            # Generate response
            model.eval()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )

            # Extract only the newly generated tokens (excluding the input prompt)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs.sequences[0][input_length:]

            # Decode only the newly generated part
            assistant_response = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            logger.info(f"\nModel Response:")
            logger.info("-" * 40)
            logger.info(assistant_response)
            logger.info("-" * 40)

            # Quick format analysis
            has_think = (
                "<think>" in assistant_response and "</think>" in assistant_response
            )
            has_boxed = "\\boxed{" in assistant_response and "}" in assistant_response
            logger.info(f"Format: Think={has_think}, Boxed={has_boxed}")

            logger.info(f"{'='*60}\n")
            model.train()


def prepare_format_dataset(examples):
    """Prepare the format alignment dataset"""
    # The dataset should already be in the correct format with messages
    # We just need to extract the text field
    texts = []

    for messages in examples["messages"]:
        # Apply chat template to the messages
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)

    return {"text": texts}


def main():
    global tokenizer  # Make tokenizer global for the prepare function

    # Model configuration - using the model from the first SFT stage
    # Should be downloaded with 
    # huggingface-cli download Looyyd/chess-sft-qwen --local-dir './chess_sft_qwen_hf/' --exclude "*00*optim_states.pt"
    model_name = "./chess_sft_qwen_hf/checkpoint-5000/"  # Output from train_sft.py

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Format alignment training for chess model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Looyyd/chess-format-alignment",
        help="Hugging Face dataset path",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="Maximum training steps"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./chess_format_aligned",
        help="Output directory for the model",
    )
    parser.add_argument(
        "--eval-split-ratio",
        type=float,
        default=0.05,
        help="Ratio of data to use for evaluation (default: 0.05)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=20,
        help="Run evaluation every N steps (default: 20)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Number of evaluation steps with no improvement before early stopping (default: 3)",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.001,
        help="Minimum change in loss to qualify as improvement (default: 0.001)",
    )
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Load dataset from Hugging Face
    logger.info(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(
        args.dataset,
        data_files={"train": "train.jsonl"},
        split="train",
    )

    # If dataset is large, optionally limit it
    if DEBUG:
        dataset = dataset.select(range(min(100, len(dataset))))

    # Split the dataset into train and eval
    logger.info(f"Splitting dataset with eval ratio: {args.eval_split_ratio}")
    split_dataset = dataset.train_test_split(
        test_size=args.eval_split_ratio, seed=42, shuffle=True  # For reproducibility
    )

    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # Load chess dataset for callback (using same dataset as training for position generation)
    chess_dataset = load_dataset(
        "Looyyd/chess-dataset",
        data_files={"train": "train.jsonl"},
        split="train",
        streaming=False,
    )
    if DEBUG:
        chess_dataset = chess_dataset.take(100)

    # Get system prompt from shared utility
    system_prompt = get_chess_system_prompt()

    # Training arguments with evaluation
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,  # Smaller batch size for format training
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        num_train_epochs=1,  # Usually one epoch is enough for format alignment
        max_steps=args.max_steps,
        learning_rate=1e-6,  # Lower learning rate to preserve capabilities
        warmup_steps=50,
        logging_steps=10,
        save_steps=args.eval_steps,
        save_total_limit=2,
        save_strategy="steps",
        eval_strategy="steps",  # Enable evaluation
        eval_steps=args.eval_steps,  # Evaluate every N steps
        metric_for_best_model="eval_loss",  # Use eval loss for model selection
        greater_is_better=False,  # Lower loss is better
        load_best_model_at_end=True,  # Load best model when training ends
        bf16=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        # Completion only, to train on what we really care about
        completion_only_loss=True,
        # SFT specific parameters
        max_length=4096,
        packing=False,
        # dataset_text_field="text",
        report_to="none" if DEBUG else "wandb",
        push_to_hub=not DEBUG,
        hub_strategy="end",
        accelerator_config={
            "dispatch_batches": False,
        },
        use_liger_kernel=True,
        # Gradient checkpointing for memory efficiency
        gradient_checkpointing=True,
    )

    # Create our simple progress callback
    progress_callback = SimpleProgressCallback(
        tokenizer=tokenizer, chess_dataset=chess_dataset, inference_steps=10
    )

    # Initialize trainer with evaluation dataset
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Add evaluation dataset
        processing_class=tokenizer,
        callbacks=[progress_callback],  # Add our progress callback
    )

    # Fine-tune the model
    logger.info("Starting format alignment training with evaluation...")
    trainer.train()

    # Save the model
    trainer.save_model()
    logger.info(f"Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
