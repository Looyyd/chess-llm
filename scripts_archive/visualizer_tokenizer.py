#!/usr/bin/env python
# coding: utf-8

import re
from transformers import AutoTokenizer
from termcolor import colored
import random


def get_color_for_index(idx):
    """Get a color for a given token index"""
    colors = ["red", "green", "yellow", "blue", "magenta", "cyan"]
    return colors[idx % len(colors)]


def visualize_tokenization(tokenizer, text, show_special_tokens=True):
    """Visualize how text is tokenized with colors"""

    # Tokenize the text
    tokens = tokenizer.tokenize(text, add_special_tokens=False)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # If we want to show special tokens, encode properly
    if show_special_tokens:
        encoded = tokenizer.encode(text, add_special_tokens=True)
        full_tokens = tokenizer.convert_ids_to_tokens(encoded)

        print("\n" + "=" * 80)
        print("TOKENIZATION WITH SPECIAL TOKENS")
        print("=" * 80)

        # Print tokens with colors
        print("\nColored tokens:")
        for i, token in enumerate(full_tokens):
            color = get_color_for_index(i)
            print(colored(f"[{i}] {repr(token)}", color), end=" ")
            if (i + 1) % 10 == 0:  # New line every 10 tokens
                print()

        print(f"\n\nTotal tokens (with special): {len(full_tokens)}")

        # Show token IDs
        print("\nToken IDs:")
        for i in range(0, len(encoded), 10):
            chunk = encoded[i : i + 10]
            print(f"[{i:3d}-{i+len(chunk)-1:3d}]: {chunk}")

    print("\n" + "=" * 80)
    print("TOKENIZATION WITHOUT SPECIAL TOKENS")
    print("=" * 80)

    # Print regular tokens with colors
    print("\nColored tokens:")
    for i, token in enumerate(tokens):
        color = get_color_for_index(i)
        print(colored(f"[{i}] {repr(token)}", color), end=" ")
        if (i + 1) % 10 == 0:  # New line every 10 tokens
            print()

    print(f"\n\nTotal tokens (without special): {len(tokens)}")

    # Reconstruct text with colors to show boundaries
    print("\n" + "=" * 80)
    print("RECONSTRUCTED TEXT WITH TOKEN BOUNDARIES")
    print("=" * 80 + "\n")

    # This is tricky because tokens might not map cleanly to text
    # We'll do our best to show the mapping
    reconstructed = ""
    for i, token in enumerate(tokens):
        color = get_color_for_index(i)
        # Clean up token representation
        token_str = token.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ")
        reconstructed += colored(token_str, color)

    print(reconstructed)


def verify_board_tokenization_consistency(tokenizer, num_samples=10):
    """Verify that different board states always tokenize to the same number of tokens"""

    print("\n" + "=" * 80)
    print("BOARD TOKENIZATION CONSISTENCY CHECK")
    print("=" * 80)

    # Template for board representation
    def generate_board_state(pieces_dict=None):
        """Generate a board state with optional piece placements"""
        # Default empty board
        board = [["." for _ in range(8)] for _ in range(8)]

        # Add some pieces if provided
        if pieces_dict:
            for pos, piece in pieces_dict.items():
                row, col = pos
                board[row][col] = piece

        # Format as in your example
        lines = []
        lines.append("  a b c d e f g h")
        lines.append("  ----------------")
        for i in range(8):
            row_num = 8 - i
            row_str = f"{row_num}| " + " ".join(board[i]) + f" |{row_num}"
            lines.append(row_str)
        lines.append("  ----------------")
        lines.append("  a b c d e f g h")

        return "\n".join(lines)

    # Test different board configurations
    test_configs = [
        # Empty board
        {},
        # Starting position
        {
            (0, 0): "r",
            (0, 1): "n",
            (0, 2): "b",
            (0, 3): "q",
            (0, 4): "k",
            (0, 5): "b",
            (0, 6): "n",
            (0, 7): "r",
            (1, 0): "p",
            (1, 1): "p",
            (1, 2): "p",
            (1, 3): "p",
            (1, 4): "p",
            (1, 5): "p",
            (1, 6): "p",
            (1, 7): "p",
            (6, 0): "P",
            (6, 1): "P",
            (6, 2): "P",
            (6, 3): "P",
            (6, 4): "P",
            (6, 5): "P",
            (6, 6): "P",
            (6, 7): "P",
            (7, 0): "R",
            (7, 1): "N",
            (7, 2): "B",
            (7, 3): "Q",
            (7, 4): "K",
            (7, 5): "B",
            (7, 6): "N",
            (7, 7): "R",
        },
        # Endgame position
        {(0, 4): "k", (2, 3): "p", (5, 4): "P", (7, 4): "K", (7, 7): "R"},
        # Random midgame
        {
            (0, 4): "k",
            (0, 7): "r",
            (1, 0): "p",
            (1, 5): "p",
            (1, 6): "p",
            (1, 7): "p",
            (2, 2): "n",
            (2, 5): "n",
            (3, 1): "p",
            (3, 4): "p",
            (4, 3): "P",
            (4, 4): "P",
            (5, 5): "N",
            (6, 0): "P",
            (6, 1): "P",
            (6, 5): "P",
            (6, 6): "P",
            (7, 4): "K",
        },
    ]

    token_counts = []

    for i, config in enumerate(test_configs):
        board_state = generate_board_state(config)
        tokens = tokenizer.tokenize(board_state, add_special_tokens=False)
        token_counts.append(len(tokens))

        print(f"\nTest board {i + 1}:")
        print(f"Number of tokens: {len(tokens)}")
        if i == 0:  # Show tokens for first board
            print("Tokens:", tokens[:20], "..." if len(tokens) > 20 else "")

    # Extract and test your actual board from the prompt
    print("\nYour actual board from the prompt:")
    actual_board = """  a b c d e f g h
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
  a b c d e f g h"""

    actual_tokens = tokenizer.tokenize(actual_board, add_special_tokens=False)
    print(f"Number of tokens: {len(actual_tokens)}")
    token_counts.append(len(actual_tokens))

    # Check consistency
    print("\n" + "-" * 40)
    print("CONSISTENCY RESULTS:")
    print(f"Token counts: {token_counts}")

    if len(set(token_counts)) == 1:
        print(
            f"✓ SUCCESS: All board states tokenize to exactly {token_counts[0]} tokens!"
        )
    else:
        print(f"✗ WARNING: Board states tokenize to different numbers of tokens!")
        print(f"  Min tokens: {min(token_counts)}")
        print(f"  Max tokens: {max(token_counts)}")
        print(f"  Variation: {max(token_counts) - min(token_counts)} tokens")

        # Analyze which parts cause variation
        print("\nAnalyzing token differences...")
        for i, count in enumerate(token_counts):
            if count != token_counts[0]:
                print(
                    f"  Board {i + 1}: {count} tokens (diff: {count - token_counts[0]})"
                )

    return len(set(token_counts)) == 1


def main():
    # Initialize tokenizer
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Print tokenizer info
    print(f"\nTokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print(f"Padding token: {repr(tokenizer.pad_token)} (id: {tokenizer.pad_token_id})")
    print(f"EOS token: {repr(tokenizer.eos_token)} (id: {tokenizer.eos_token_id})")
    print(f"BOS token: {repr(tokenizer.bos_token)} (id: {tokenizer.bos_token_id})")

    # Create the test prompt (from your code)
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

    # Apply chat template
    chat_formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print("\n" + "=" * 80)
    print("CHAT TEMPLATE OUTPUT")
    print("=" * 80)
    print(chat_formatted[:500] + "..." if len(chat_formatted) > 500 else chat_formatted)

    # Visualize tokenization of the chat-formatted prompt
    visualize_tokenization(tokenizer, chat_formatted, show_special_tokens=True)

    # Also show tokenization of just the chess position part
    print("\n\n" + "=" * 80)
    print("CHESS POSITION ONLY (for comparison)")
    print("=" * 80)
    visualize_tokenization(tokenizer, test_position, show_special_tokens=False)

    # Show some interesting statistics
    print("\n" + "=" * 80)
    print("TOKENIZATION STATISTICS")
    print("=" * 80)

    # Analyze move history tokenization
    move_history = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3 c6a5 b3c2 c7c5 d2d4 d8c7 b1d2 c5d4 c3d4 a5c6"
    move_tokens = tokenizer.tokenize(move_history)
    print(f"\nMove history tokens: {len(move_tokens)}")
    print("Sample move tokens:", move_tokens[:10])

    # Analyze board representation tokenization
    board_lines = test_position.split("\n")[8:17]  # Extract just the board
    board_text = "\n".join(board_lines)
    board_tokens = tokenizer.tokenize(board_text)
    print(f"\nBoard representation tokens: {len(board_tokens)}")

    # Show how individual chess moves are tokenized
    print("\n" + "=" * 80)
    print("INDIVIDUAL MOVE TOKENIZATION")
    print("=" * 80)

    sample_moves = ["e2e4", "Nf3", "O-O", "Qxd4", "a7a6", "h2h3"]
    for move in sample_moves:
        tokens = tokenizer.tokenize(move)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        print(f"\n{move:8} -> tokens: {tokens} -> ids: {token_ids}")

    # After the existing analysis, add:
    consistent = verify_board_tokenization_consistency(tokenizer)
    print(f"Board tokenization consistency: {consistent}")

    if not consistent:
        print("\n" + "=" * 80)
        print("TOKENIZATION FIX SUGGESTIONS")
        print("=" * 80)
        print("\nTo ensure consistent tokenization, consider:")
        print("1. Using a fixed-width representation with padding")
        print("2. Encoding the board as a FEN string instead")
        print("3. Using a custom encoding scheme (e.g., base64 of board state)")
        print("4. Pre-tokenizing and padding to a fixed length")


if __name__ == "__main__":
    # Install required package if not present
    try:
        from termcolor import colored
    except ImportError:
        print("Installing termcolor for colored output...")
        import subprocess

        subprocess.check_call(["pip", "install", "termcolor"])
        from termcolor import colored

    main()
