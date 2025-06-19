#!/usr/bin/env python
# coding: utf-8

import json
import os
from datasets import load_dataset
from openai import OpenAI
from utils.chess_utils import board_to_grid
from utils.dataset_utils import select_weighted_position, reconstruct_board_position
import chess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_checklist(phase):
    """Load checklist markdown file for a specific game phase"""
    checklist_path = os.path.join(
        os.path.dirname(__file__), "checklists", f"{phase}.md"
    )
    try:
        with open(checklist_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Checklist file not found: {checklist_path}")
        return ""


# Cache all checklists at module level to avoid repeated file I/O
ALL_CHECKLISTS = f"""
{load_checklist("opening")}

---

{load_checklist("midgame")}

---

{load_checklist("endgame")}
"""

# Create the system prompt (same as in train_rl.py)
system_prompt = f"""You are a chess engine. Given a chess position, analyze the position and determine the best move.

First, analyze the position inside <think> tags, using the following checklists to guide your thinking:

{ALL_CHECKLISTS}

Choose the most appropriate checklist(s) based on the game phase and work through them systematically as you analyze the position. Then provide your chosen move in UCI format inside \\boxed{{}} tags.

Example format:
<think>
The position appears to be in the [opening/middlegame/endgame] phase. Following the relevant checklist:

1. Safety & Basic Tactics:
- My king is safe on g1, not in check
- No pieces are hanging
- No immediate captures available...

2. [Continue through the relevant checklist sections...]

Based on this analysis, the best move is...
</think>
\\boxed{{f3e5}}"""


def generate_chess_prompt(moves, position_idx):
    """Generate a chess prompt in the same format as train_rl.py"""
    # Reconstruct board position
    board, move_history, move_history_str = reconstruct_board_position(
        moves, position_idx
    )

    # Determine whose turn it is
    turn = "White" if board.turn == chess.WHITE else "Black"

    # Create board visualization
    board_grid = board_to_grid(board)

    user_prompt = f"""Current game position:

Move history (UCI format): {move_history_str}
Turn: {turn}

Current board state:
{board_grid}

What is the best move? Analyze the position and provide your answer."""

    return user_prompt, board, move_history_str, turn


def get_gpt_response(system_prompt, user_prompt):
    """Get response from GPT-4.1 mini"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting GPT response: {e}")
        return None


def process_single_example(example_data, example_idx, target_count, print_lock):
    """Process a single chess example and return the result"""
    i, example = example_data
    moves = example["moves"]

    # Skip if game is too short (same logic as train_rl.py)
    if len(moves) < 8:
        return None

    # Select weighted position
    position_idx = select_weighted_position(moves)

    try:
        # Generate prompt
        user_prompt, board, move_history_str, turn = generate_chess_prompt(
            moves, position_idx
        )

        logger.info(f"Processing example {i}, position: {move_history_str[:30]}...")

        # Get GPT response
        start_time = time.time()
        gpt_response = get_gpt_response(system_prompt, user_prompt)
        response_time = time.time() - start_time

        if gpt_response:
            example_data = {
                "system": system_prompt,
                "user": user_prompt,
                "assistant": gpt_response,
                "move_history": move_history_str,
                "turn": turn,
                "position_idx": position_idx,
                "original_game_moves": moves,
            }

            # Use lock for thread-safe printing
            with print_lock:
                print(f"\n{'='*60}")
                print(
                    f"EXAMPLE {example_idx + 1}/{target_count} (processed in {response_time:.2f}s)"
                )
                print(f"{'='*60}")
                print(f"Position: {move_history_str}")
                print(f"Turn: {turn}")
                print(f"\nBoard:")
                print(board_to_grid(board))
                print(f"\nGPT Response:")
                print(gpt_response)
                print(f"{'='*60}\n")

            return example_data
        else:
            logger.warning(f"Failed to get GPT response for example {i}")
            return None

    except Exception as e:
        logger.error(f"Error processing example {i}: {e}")
        return None


def main():
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Please set OPENAI_API_KEY environment variable")
        return

    # Load dataset from Hugging Face (same as train_rl.py)
    logger.info("Loading dataset from Hugging Face...")
    dataset = load_dataset(
        "Looyyd/chess-dataset",
        data_files={"train": "train.jsonl"},
        split="train",
        streaming=False,
    )

    target_count = 1000
    dataset = dataset.take(2000)

    max_workers = 20  # Number of parallel API calls

    # Create a lock for thread-safe printing
    print_lock = Lock()

    # Convert dataset to list for easier parallel processing
    dataset_list = list(enumerate(dataset))

    generated_examples = []

    logger.info(f"Starting parallel processing with {max_workers} workers...")
    start_time = time.time()

    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_example = {
            executor.submit(
                process_single_example,
                example_data,
                len(generated_examples),
                target_count,
                print_lock,
            ): example_data
            for example_data in dataset_list
        }

        # Process completed tasks
        for future in as_completed(future_to_example):
            if len(generated_examples) >= target_count:
                # Cancel remaining futures
                for f in future_to_example:
                    if not f.done():
                        f.cancel()
                break

            result = future.result()
            if result:
                generated_examples.append(result)
                logger.info(
                    f"Completed {len(generated_examples)}/{target_count} examples"
                )

    total_time = time.time() - start_time
    logger.info(
        f"Generated {len(generated_examples)} examples in {total_time:.2f} seconds"
    )
    logger.info(
        f"Average time per example: {total_time/len(generated_examples):.2f} seconds"
    )

    # Save the dataset
    output_file = "format_training_dataset.jsonl"
    with open(output_file, "w") as f:
        for example in generated_examples:
            f.write(json.dumps(example) + "\n")

    logger.info(
        f"Generated {len(generated_examples)} examples and saved to {output_file}"
    )

    # Also save in Hugging Face dataset format for easy loading
    hf_format = []
    for example in generated_examples:
        hf_format.append(
            {
                "messages": [
                    {"role": "system", "content": example["system"]},
                    {"role": "user", "content": example["user"]},
                    {"role": "assistant", "content": example["assistant"]},
                ]
            }
        )

    with open("format_training_dataset_hf.jsonl", "w") as f:
        for example in hf_format:
            f.write(json.dumps(example) + "\n")

    logger.info("Also saved in Hugging Face format as format_training_dataset_hf.jsonl")


if __name__ == "__main__":
    main()
