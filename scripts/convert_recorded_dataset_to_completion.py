#!/usr/bin/env python
# coding: utf-8

import json
import os
import logging
import argparse
from tqdm import tqdm

# Import shared utilities - you'll need to ensure these are in the correct path
from utils.prompt_utils import (
    get_chess_system_prompt,
    create_chess_user_prompt,
    format_chess_messages,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_prompt_completion(entry):
    """
    Convert a manual chess dataset entry to prompt-completion format

    Args:
        entry: Dict containing chess analysis data

    Returns:
        Dict with prompt and completion fields, or None if entry is invalid
    """
    # Skip invalid entries
    if entry.get("invalid", False):
        return None

    # Extract required fields
    board_grid = entry.get("board_grid", "")
    move_history = entry.get("move_history", "")
    turn = entry.get("turn", "White")
    transcript_formatted = entry.get("transcript_formatted", "")
    final_move = entry.get("final_move", "")

    # Skip entries without required data
    if not all([board_grid, transcript_formatted, final_move]):
        logger.warning(
            f"Skipping entry due to missing data: board_grid={bool(board_grid)}, "
            f"transcript={bool(transcript_formatted)}, move={bool(final_move)}"
        )
        return None

    # Get system prompt
    system_prompt = get_chess_system_prompt()

    # Create user prompt
    user_prompt = create_chess_user_prompt(board_grid, move_history, turn)

    # Format as messages
    messages = format_chess_messages(system_prompt, user_prompt)

    # Create the completion in the expected format with thinking tags
    completion = f"""<think>
{transcript_formatted}
</think>
\\boxed{{{final_move}}}"""

    # Add the assistant's response to messages
    messages.append({"role": "assistant", "content": completion})

    # Create the output entry
    output_entry = {
        "prompt": messages[:-1],  # Everything except the assistant's response
        "completion": [messages[-1]],  # Just the assistant's response
        # Include original metadata for reference
        "metadata": {
            "board_fen": entry.get("board_fen", ""),
            "position_idx": entry.get("position_idx", 0),
            "timestamp": entry.get("timestamp", ""),
            "turn": turn,
            "final_move": final_move,
        },
    }

    return output_entry


def process_dataset(input_file, output_file, max_entries=None):
    """
    Process the manual chess dataset and convert to prompt-completion format

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        max_entries: Maximum number of entries to process (None for all)
    """

    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} does not exist")
        return

    # Count total entries for progress bar
    total_entries = 0
    with open(input_file, "r") as f:
        for _ in f:
            total_entries += 1

    if max_entries:
        total_entries = min(total_entries, max_entries)

    processed_count = 0
    skipped_count = 0

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:

        pbar = tqdm(total=total_entries, desc="Converting dataset")

        for line_num, line in enumerate(infile, 1):
            if max_entries and processed_count >= max_entries:
                break

            try:
                entry = json.loads(line.strip())

                # Convert to prompt-completion format
                converted_entry = convert_to_prompt_completion(entry)

                if converted_entry:
                    # Write the converted entry
                    outfile.write(json.dumps(converted_entry) + "\n")
                    processed_count += 1
                else:
                    skipped_count += 1

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error at line {line_num}: {e}")
                skipped_count += 1
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
                skipped_count += 1

            pbar.update(1)

        pbar.close()

    logger.info(f"Conversion complete!")
    logger.info(f"Total processed: {processed_count}")
    logger.info(f"Total skipped: {skipped_count}")
    logger.info(f"Output saved to: {output_file}")


def preview_conversion(input_file, entry_index=0):
    """
    Preview the conversion of a specific entry

    Args:
        input_file: Path to input JSONL file
        entry_index: Index of entry to preview (0-based)
    """

    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            if i == entry_index:
                entry = json.loads(line.strip())

                print(f"\n{'='*60}")
                print(f"ORIGINAL ENTRY:")
                print(f"{'='*60}")
                print(f"Analysis: {entry.get('analysis', '')[:100]}...")
                print(f"Formatted: {entry.get('transcript_formatted', '')[:100]}...")
                print(f"Final move: {entry.get('final_move', '')}")
                print(f"Turn: {entry.get('turn', '')}")

                # Convert the entry
                converted = convert_to_prompt_completion(entry)

                if converted:
                    print(f"\n{'='*60}")
                    print(f"CONVERTED ENTRY:")
                    print(f"{'='*60}")

                    # Show the prompt (system + user messages)
                    print("\nPROMPT MESSAGES:")
                    for msg in converted["prompt"]:
                        print(f"\nRole: {msg['role']}")
                        print(
                            f"Content: {msg['content'][:200]}..."
                            if len(msg["content"]) > 200
                            else f"Content: {msg['content']}"
                        )

                    # Show the completion
                    print("\nCOMPLETION:")
                    completion = converted["completion"][0]
                    print(f"Role: {completion['role']}")
                    print(f"Content: {completion['content']}")

                    # Show metadata
                    print(f"\nMETADATA: {converted['metadata']}")
                else:
                    print("\nEntry could not be converted!")

                return

        print(f"Entry index {entry_index} not found")


def validate_output(output_file, num_samples=5):
    """
    Validate the output file by checking a few random samples

    Args:
        output_file: Path to output JSONL file
        num_samples: Number of samples to check
    """

    import random

    # Load all entries
    entries = []
    with open(output_file, "r") as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    if not entries:
        logger.error("No entries found in output file")
        return

    # Sample random entries
    samples = random.sample(entries, min(num_samples, len(entries)))

    logger.info(f"\nValidating {len(samples)} random samples from output...")

    for i, entry in enumerate(samples, 1):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i}:")
        print(f"{'='*60}")

        # Check structure
        has_prompt = "prompt" in entry
        has_completion = "completion" in entry
        has_metadata = "metadata" in entry

        print(f"Has prompt: {has_prompt}")
        print(f"Has completion: {has_completion}")
        print(f"Has metadata: {has_metadata}")

        if has_prompt:
            print(f"Prompt messages: {len(entry['prompt'])}")

        if has_completion:
            completion_content = entry["completion"][0]["content"]
            has_think_tags = (
                "<think>" in completion_content and "</think>" in completion_content
            )
            has_boxed_move = (
                "\\boxed{" in completion_content and "}" in completion_content
            )

            print(f"Has think tags: {has_think_tags}")
            print(f"Has boxed move: {has_boxed_move}")

            # Extract move from boxed notation
            if has_boxed_move:
                start = completion_content.find("\\boxed{") + 7
                end = completion_content.find("}", start)
                move = completion_content[start:end]
                print(f"Extracted move: {move}")

        if has_metadata:
            print(f"Metadata: {entry['metadata']}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert manual chess dataset to prompt-completion format"
    )
    parser.add_argument(
        "input_file", help="Input JSONL file path (manual chess dataset)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (defaults to input_file with _prompt_completion suffix)",
    )
    parser.add_argument(
        "--max-entries", "-m", type=int, help="Maximum entries to process (for testing)"
    )
    parser.add_argument(
        "--preview",
        "-p",
        type=int,
        help="Preview conversion of specific entry (0-based index)",
    )
    parser.add_argument(
        "--validate",
        "-v",
        action="store_true",
        help="Validate the output file after conversion",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=5,
        help="Number of samples to validate (default: 5)",
    )

    args = parser.parse_args()

    # Determine output file
    if args.output is None:
        base_name = os.path.splitext(args.input_file)[0]
        output_file = f"{base_name}_prompt_completion.jsonl"
    else:
        output_file = args.output

    # Handle different modes
    if args.preview is not None:
        preview_conversion(args.input_file, args.preview)
    else:
        # Process the dataset
        process_dataset(args.input_file, output_file, args.max_entries)

        # Optionally validate the output
        if args.validate:
            validate_output(output_file, args.validation_samples)


if __name__ == "__main__":
    main()
