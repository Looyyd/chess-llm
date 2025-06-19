#!/usr/bin/env python
# coding: utf-8

import sys
import os

# Add parent directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import chess
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from utils.chess_utils import board_to_grid
from utils.dataset_utils import select_weighted_position, reconstruct_board_position
from utils.prompt_utils import (
    get_chess_system_prompt,
    create_chess_user_prompt,
    format_chess_messages,
)


def test_base_model_inference(model_path="Qwen/Qwen2.5-7B-Instruct", num_samples=5):
    """Test the base Qwen model on random chess positions"""

    print(f"Loading model and tokenizer from {model_path}...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Get system prompt
    system_prompt = get_chess_system_prompt()

    print("Loading chess dataset...")
    # Load dataset from Hugging Face
    dataset = load_dataset(
        "Looyyd/chess-dataset",
        data_files={"train": "train.jsonl"},
        split="train",
        streaming=False,
    )

    # Take a sample
    dataset = dataset.take(1000)

    print(f"\nTesting base model on {num_samples} random chess positions...")
    print("=" * 80)

    for i in range(num_samples):
        print(f"\n--- SAMPLE {i+1}/{num_samples} ---")

        # Select a random game
        game_idx = random.randint(0, len(dataset) - 1)
        game = dataset[game_idx]
        moves = game["moves"]

        # Skip if game is too short
        if len(moves) < 8:
            continue

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
        messages = format_chess_messages(system_prompt, user_prompt)

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print(f"Position after move {position_idx} ({turn}'s turn)")
        print(f"Move history: {move_history_str}")
        print("\nBoard:")
        print(board_grid)

        # Can prefill with <think> tag to encourage thinking format
        # prefill_text = "<think>"
        # prompt_with_prefill = prompt + prefill_text
        prompt_with_prefill = prompt

        # Generate response
        inputs = tokenizer(prompt_with_prefill, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        # Extract only the newly generated tokens (excluding the input prompt)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs.sequences[0][input_length:]
        
        # Decode only the newly generated part
        assistant_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print("\nMODEL RESPONSE:")
        print("-" * 40)
        print(assistant_response)
        print("-" * 40)

        # Check if response has thinking tags and boxed move
        has_think = "<think>" in assistant_response and "</think>" in assistant_response
        has_boxed = "\\boxed{" in assistant_response and "}" in assistant_response
        print(f"\nFormat Analysis:")
        print(f"  Has <think> tags: {has_think}")
        print(f"  Has \\boxed{{}} move: {has_boxed}")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test base Qwen model on chess positions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model path or name (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of random positions to test (default: 5)",
    )

    args = parser.parse_args()

    test_base_model_inference(args.model, args.samples)


if __name__ == "__main__":
    main()
