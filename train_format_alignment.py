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
from accelerate import PartialState, Accelerator
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

# Initialize accelerator for logging control
accelerator = Accelerator()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG = False


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
    # model_name = "./chess_sft_qwen/checkpoint-2500/"  # Output from train_sft.py
    model_name = "Looyyd/chess-sft-qwen"  # Output from train_sft.py

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
        default=0.08,
        help="Ratio of data to use for evaluation (default: 0.08)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=75,
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_size="right")
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # prevents an error ?
        attn_implementation="flash_attention_2",
    )

    # Load dataset from Hugging Face
    if accelerator.is_main_process:
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
    if accelerator.is_main_process:
        logger.info(f"Splitting dataset with eval ratio: {args.eval_split_ratio}")
    split_dataset = dataset.train_test_split(
        test_size=args.eval_split_ratio, seed=42, shuffle=True  # For reproducibility
    )

    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    if accelerator.is_main_process:
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # Training arguments with evaluation
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,  # Smaller batch size for format training
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        num_train_epochs=1,  # Usually one epoch is enough for format alignment
        max_steps=args.max_steps,
        learning_rate=1e-2,  # Lower learning rate to preserve capabilities
        warmup_steps=50,
        logging_steps=10,
        save_steps=args.eval_steps * 2,
        save_total_limit=2,
        save_strategy="steps",
        eval_strategy="steps",  # Enable evaluation
        eval_steps=args.eval_steps,  # Evaluate every N steps
        metric_for_best_model="eval_loss",  # Use eval loss for model selection
        greater_is_better=False,  # Lower loss is better
        load_best_model_at_end=True,  # Load best model when training ends
        bf16=True,
        optim="adamw_torch",
        max_grad_norm=1e-8,
        ddp_find_unused_parameters=False,
        # Completion only, to train on what we really care about
        completion_only_loss=True,
        # SFT specific parameters
        max_length=4096,
        packing=False,
        # dataset_text_field="text",
        report_to="none" if DEBUG else "wandb",
        push_to_hub=False,
        hub_strategy="end",
        accelerator_config={
            "dispatch_batches": False,
        },
        use_liger_kernel=True,
        # Gradient checkpointing for memory efficiency
        gradient_checkpointing=True,
    )

    # Initialize trainer with evaluation dataset
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Add evaluation dataset
        processing_class=tokenizer,
    )

    # Fine-tune the model
    if accelerator.is_main_process:
        logger.info("Starting format alignment training with evaluation...")
    trainer.train()

    # Save the model
    trainer.save_model()
    if accelerator.is_main_process:
        logger.info(f"Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
