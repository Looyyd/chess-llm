#!/usr/bin/env python
# coding: utf-8

import random
import chess
from utils.chess_utils import board_to_grid
from utils.dataset_utils import load_lichess_dataset, select_weighted_position, reconstruct_board_position




def show_random_game_position():
    """Load dataset and show a random game position in prompt format"""

    # Load dataset
    print("Loading dataset...")
    dataset = load_lichess_dataset(
        "./data/lichess_2013_12_compact.jsonl",
        split="train",
        streaming=True
    )

    # Select a random game
    dataset = dataset.shuffle()
    # For streaming dataset, we need to iterate and pick a random one
    # Skip a random number of games to get a random game
    skip_count = random.randint(0, 10)

    game = None
    for i, example in enumerate(dataset):
        if i == skip_count:
            game = example
            break
    moves = game["moves"]

    print(f"\nSelected game with {len(moves)} moves")
    print(f"Full game moves: {' '.join(moves)}")

    # Select a random position from the game (weighted towards later positions)
    position_idx = select_weighted_position(moves)
    
    print(f"\nSelected position after move {position_idx}")
    
    # Reconstruct board position
    board, move_history, move_history_str = reconstruct_board_position(moves, position_idx)
    
    # Determine whose turn it is
    turn = "White" if board.turn == chess.WHITE else "Black"

    # Create board visualization
    board_grid = board_to_grid(board)

    # Create the user prompt (same format as in training script)
    user_prompt = f"""Current game position:

Move history (UCI format): {move_history_str}
Turn: {turn}

Current board state:
{board_grid}

What is the best move? Analyze the position and provide your answer."""

    print("\n" + "=" * 80)
    print("USER PROMPT:")
    print("=" * 80)
    print(user_prompt)
    print("=" * 80)


if __name__ == "__main__":
    show_random_game_position()
