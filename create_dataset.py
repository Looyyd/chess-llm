import chess.pgn
import json
from pathlib import Path


def convert_pgn_to_jsonl(input_pgn_path, output_jsonl_path, games_per_chunk=10000):
    """
    Convert a PGN file to JSONL format, keeping only essential data.
    Each line contains one complete game. Easier to parse and slightly less storage.
    """
    input_path = Path(input_pgn_path)
    output_path = Path(output_jsonl_path)
    games_processed = 0
    games_written = 0

    # Batch writing
    batch_size = 10000  # Write every batch_size games
    batch = []

    with open(input_path, "r", encoding="utf-8") as pgn_file, open(
        output_path, "w", encoding="utf-8"
    ) as out_file:
        while True:
            # Read one game at a time - memory efficient
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games_processed += 1
            moves = list(game.mainline_moves())
            try:
                # Extract only what we need
                game_data = {
                    "white_elo": int(game.headers.get("WhiteElo", 0)),
                    "black_elo": int(game.headers.get("BlackElo", 0)),
                    "time_control": game.headers.get("TimeControl", "0"),
                    "opening": game.headers.get("Opening", "Unknown"),
                    "eco": game.headers.get("ECO", ""),
                    "moves": [
                        move.uci() for move in moves
                    ],  # UCI format is more compact
                }

                # Add to batch instead of writing immediately
                batch.append(json.dumps(game_data) + "\n")
                games_written += 1

                # Write batch when it's full
                if len(batch) >= batch_size:
                    out_file.writelines(batch)
                    batch = []

                if games_processed % games_per_chunk == 0:
                    print(f"Processed {games_processed} games, kept {games_written}")

            except Exception as e:
                # Skip games with parsing errors
                print(f"Error processing game {games_processed}: {e}")
                continue

        # Write any remaining games in the batch
        if batch:
            out_file.writelines(batch)

    print(
        f"Conversion complete! Processed {games_processed} games, wrote {games_written}"
    )
    return games_written


if __name__ == "__main__":
    convert_pgn_to_jsonl(
        "./data/lichess_db_standard_rated_2014-01.pgn",
        "./data/lichess_2014_01_compact.jsonl",
    )
