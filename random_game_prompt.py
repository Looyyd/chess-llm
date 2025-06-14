#!/usr/bin/env python
# coding: utf-8

import random
import chess
from datasets import load_dataset


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


def show_random_game_position():
    """Load dataset and show a random game position in prompt format"""

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files="./data/lichess_2013_12_compact.jsonl",
        split="train",
        streaming=True,
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
    num_positions = len(moves) - 1

    # Create weights that linearly increase from 1 to 5
    weights = []
    for pos in range(num_positions):
        if num_positions == 1:
            weight = 1.0
        else:
            weight = 1.0 + 4.0 * (pos / (num_positions - 1))
        weights.append(weight)

    # Select position using weights
    position_idx = random.choices(range(num_positions), weights=weights, k=1)[0]

    print(f"\nSelected position after move {position_idx}")

    # Reconstruct board up to that position
    board = chess.Board()
    move_history = []

    for j in range(position_idx):
        move = moves[j]
        board.push_uci(move)
        move_history.append(move)

    # Determine whose turn it is
    turn = "White" if board.turn == chess.WHITE else "Black"

    # Format move history
    move_history_str = " ".join(move_history) if move_history else "Game start"

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
