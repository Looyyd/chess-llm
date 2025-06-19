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
