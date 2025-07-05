#!/usr/bin/env python
# coding: utf-8

import json
import os
import logging
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import argparse
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI()


class ChessTranscriptFormatted(BaseModel):
    """Schema for formatted chess transcript with reasoning trace and final move"""

    reasoning_trace: str
    final_move: Optional[str] = None  # UCI format like e2e4, d5d6, etc.


def format_chess_transcript(
    raw_transcript: str, board_fen: str, turn: str
) -> ChessTranscriptFormatted:
    """
    Use GPT-4o with structured outputs to format a chess transcript

    Args:
        raw_transcript: The raw voice-to-text transcript
        board_fen: The FEN position of the chess board
        turn: Whose turn it is (White/Black)

    Returns:
        ChessTranscriptFormatted object with cleaned reasoning and final move
    """

    system_prompt = """You are a chess transcript formatter. Your job is to clean up voice-to-text transcripts of chess analysis while preserving the authentic reasoning process.

IMPORTANT RULES:
1. Fix obvious voice recognition errors (e.g., "night" → "knight", "bishop" → "bishop", "pond" → "pawn")
2. PRESERVE ALL HESITATIONS, BACKTRACKS, AND CORRECTIONS - these are valuable for the dataset
3. Keep the natural flow of thought, including "um", "uh", "wait", "actually", etc.
4. Do NOT add analysis that wasn't in the original transcript
5. Do NOT remove uncertainty or changes of mind
6. Format into readable paragraphs but maintain the original reasoning structure

For the final_move field:
- Extract the final move recommendation in UCI format (e.g., e2e4, g1f3, a7a5)
- Only include if the speaker clearly recommends a specific move
- If no clear final move is stated, leave as null
- Convert from algebraic notation if needed (e4 → e2e4, Nf3 → g1f3, etc.)

The transcript represents analysis of a chess position where it's {turn}'s turn to move."""

    user_prompt = f"""Clean up this chess analysis transcript:

Position: {board_fen}
Turn: {turn}
Raw transcript: "{raw_transcript}"

Format the reasoning trace and extract the final move recommendation."""

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_prompt.format(turn=turn)},
                {"role": "user", "content": user_prompt},
            ],
            response_format=ChessTranscriptFormatted,
            temperature=0.1,  # Low temperature for consistent formatting
        )

        return response.choices[0].message.parsed

    except Exception as e:
        logger.error(f"Error formatting transcript: {e}")
        # Return original transcript if formatting fails
        return ChessTranscriptFormatted(reasoning_trace=raw_transcript, final_move=None)


def process_dataset(input_file: str, output_file: str = None, max_entries: int = None):
    """
    Process the chess dataset and add formatted transcripts

    Args:
        input_file: Path to the input JSONL file
        output_file: Path to output file (defaults to input_file with _formatted suffix)
        max_entries: Maximum number of entries to process (for testing)
    """

    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_formatted.jsonl"

    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} does not exist")
        return

    # Count total entries for progress bar
    total_entries = 0
    with open(input_file, "r") as f:
        for _ in f:
            total_entries += 1

    processed_count = 0
    error_count = 0

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:

        pbar = tqdm(
            total=min(total_entries, max_entries or total_entries),
            desc="Processing transcripts",
        )

        for line_num, line in enumerate(infile, 1):
            if max_entries and processed_count >= max_entries:
                break

            try:
                entry = json.loads(line.strip())

                # Skip invalid entries
                if entry.get("invalid", False):
                    logger.info(f"Skipping invalid entry at line {line_num}")
                    continue

                # Skip if already processed
                if "transcript_formatted" in entry:
                    logger.info(f"Skipping already processed entry at line {line_num}")
                    # Write the existing entry as-is
                    outfile.write(json.dumps(entry) + "\n")
                    continue

                # Get the raw transcript
                raw_analysis = entry.get("analysis", "")
                if not raw_analysis.strip():
                    logger.warning(f"Empty analysis at line {line_num}")
                    # Add empty formatted fields
                    entry["transcript_formatted"] = ""
                    entry["final_move"] = None
                    outfile.write(json.dumps(entry) + "\n")
                    continue

                # Format the transcript
                logger.info(
                    f"Processing entry {processed_count + 1}: {raw_analysis[:50]}..."
                )

                formatted_result = format_chess_transcript(
                    raw_transcript=raw_analysis,
                    board_fen=entry.get("board_fen", ""),
                    turn=entry.get("turn", "White"),
                )

                # Add formatted fields to entry
                entry["transcript_formatted"] = formatted_result.reasoning_trace
                entry["final_move"] = formatted_result.final_move
                entry["formatting_processed"] = True

                # Write updated entry
                outfile.write(json.dumps(entry) + "\n")
                processed_count += 1

                logger.info(f"Processed entry {processed_count}")
                if formatted_result.final_move:
                    logger.info(f"  Final move: {formatted_result.final_move}")

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error at line {line_num}: {e}")
                error_count += 1
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
                error_count += 1
                # Write original entry on error
                try:
                    entry = json.loads(line.strip())
                    entry["transcript_formatted"] = entry.get("analysis", "")
                    entry["final_move"] = None
                    entry["formatting_error"] = str(e)
                    outfile.write(json.dumps(entry) + "\n")
                except:
                    pass

            pbar.update(1)

        pbar.close()

    logger.info(f"Processing complete!")
    logger.info(f"Total processed: {processed_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Output saved to: {output_file}")


def preview_formatting(input_file: str, entry_index: int = 0):
    """
    Preview the formatting of a specific entry without saving

    Args:
        input_file: Path to the input JSONL file
        entry_index: Index of entry to preview (0-based)
    """

    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            if i == entry_index:
                entry = json.loads(line.strip())

                if entry.get("invalid", False):
                    print("This entry is marked as invalid")
                    return

                raw_analysis = entry.get("analysis", "")
                print(f"Original transcript:")
                print(f"'{raw_analysis}'")
                print("\n" + "=" * 50 + "\n")

                formatted_result = format_chess_transcript(
                    raw_transcript=raw_analysis,
                    board_fen=entry.get("board_fen", ""),
                    turn=entry.get("turn", "White"),
                )

                print(f"Formatted reasoning trace:")
                print(f"'{formatted_result.reasoning_trace}'")
                print(f"\nFinal move: {formatted_result.final_move}")
                return

        print(f"Entry index {entry_index} not found")


def main():
    parser = argparse.ArgumentParser(
        description="Format chess transcript dataset using GPT-4o"
    )
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument("--output", "-o", help="Output file path (optional)")
    parser.add_argument(
        "--max-entries", "-m", type=int, help="Maximum entries to process (for testing)"
    )
    parser.add_argument(
        "--preview",
        "-p",
        type=int,
        help="Preview formatting of specific entry (0-based index)",
    )

    args = parser.parse_args()

    if args.preview is not None:
        preview_formatting(args.input_file, args.preview)
    else:
        process_dataset(args.input_file, args.output, args.max_entries)


if __name__ == "__main__":
    main()
