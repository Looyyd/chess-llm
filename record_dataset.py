#!/usr/bin/env python
# coding: utf-8

import json
import os
import chess
import chess.engine
import chess.svg
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from datasets import load_dataset
from utils.dataset_utils import select_weighted_position, reconstruct_board_position
from utils.chess_utils import board_to_grid
import logging
import tempfile
import numpy as np
from datetime import datetime
import soundfile as sf
from pathlib import Path
from openai import OpenAI
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
# STOCKFISH_PATH = r"/usr/games/stockfish"  # Update this path
STOCKFISH_PATH = r"C:\Users\filip\dev\stockfish\stockfish-windows-x86-64-avx2.exe"
STOCKFISH_TIME_LIMIT = 2
STOCKFISH_DEPTH = 20
OUTPUT_DIR = "manual_chess_dataset"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio_recordings")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Make sure to set this

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Global variables
current_position_data = None
dataset = None
dataset_iter = None
engine = None
last_entry_id = None
openai_client = None


def init_resources():
    """Initialize Stockfish and OpenAI"""
    global engine, dataset, dataset_iter, openai_client

    try:
        # Initialize Stockfish
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        logger.info("Stockfish initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Stockfish: {e}")
        raise

    # Initialize OpenAI
    if OPENAI_API_KEY:
        try:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    else:
        logger.error(
            "OPENAI_API_KEY not set. OpenAI transcription and formatting will not be available"
        )
        raise ValueError("OPENAI_API_KEY is required")

    # Load chess dataset
    logger.info("Loading chess dataset...")
    dataset = load_dataset(
        "Looyyd/chess-dataset",
        data_files={"train": "train.jsonl"},
        split="train",
        streaming=True,  # Use streaming for efficiency
    )
    dataset_iter = iter(dataset)


def get_top_moves(board, n=3):
    """Get top N moves from Stockfish with evaluations"""
    try:
        # Get multiple principal variations
        result = engine.analyse(
            board,
            chess.engine.Limit(time=STOCKFISH_TIME_LIMIT, depth=STOCKFISH_DEPTH),
            multipv=n,
        )

        top_moves = []
        for i in range(min(n, len(result))):
            info = result[i]
            move = info["pv"][0]
            score = info["score"].relative

            # Convert score to human-readable format
            if score.is_mate():
                eval_str = f"M{score.mate()}"
            else:
                eval_str = f"{score.score() / 100:.2f}"

            top_moves.append(
                {
                    "move": move.uci(),
                    "from": chess.square_name(move.from_square),
                    "to": chess.square_name(move.to_square),
                    "evaluation": eval_str,
                    "san": board.san(move),
                }
            )

        return top_moves
    except Exception as e:
        logger.error(f"Error getting top moves: {e}")
        return []


def create_pgn_from_position(board, move_history_str):
    """Create a PGN string from the current position"""
    import chess.pgn
    import io

    # Create a game with the moves
    game = chess.pgn.Game()

    # Add basic headers
    game.headers["Event"] = "Training Position"
    game.headers["Site"] = "Chess Dataset Creator"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "?"
    game.headers["Black"] = "?"

    # Replay moves to get to current position
    if move_history_str:
        moves = move_history_str.split()
        node = game
        board_temp = chess.Board()
        for move_uci in moves:
            try:
                move = chess.Move.from_uci(move_uci)
                node = node.add_variation(move)
                board_temp.push(move)
            except:
                pass

    # Convert to PGN string
    pgn_io = io.StringIO()
    exporter = chess.pgn.FileExporter(pgn_io)
    game.accept(exporter)

    return pgn_io.getvalue()


def get_next_position():
    """Get the next chess position from the dataset"""
    global current_position_data, dataset_iter

    while True:
        try:
            example = next(dataset_iter)
            moves = example["moves"]

            # Skip if game is too short
            if len(moves) < 8:
                continue

            # Select weighted position
            position_idx = select_weighted_position(moves)

            # Reconstruct board position
            board, move_history, move_history_str = reconstruct_board_position(
                moves, position_idx
            )

            # Get top moves from Stockfish
            top_moves = get_top_moves(board)

            # Create SVG of the board with arrows for top moves
            arrows = []
            colors = ["green", "yellow", "orange"]  # Different colors for 1st, 2nd, 3rd
            for i, move_data in enumerate(top_moves[:3]):
                move = chess.Move.from_uci(move_data["move"])
                arrows.append(
                    chess.svg.Arrow(move.from_square, move.to_square, color=colors[i])
                )

            # TODO: trying without arrows, because it biases the analysis and makes it less "human"
            arrows=[]
            board_svg = chess.svg.board(board=board, arrows=arrows, size=600)

            # Create PGN for easy import to Lichess
            pgn_string = create_pgn_from_position(board, move_history_str)

            current_position_data = {
                "board": board,
                "board_fen": board.fen(),
                "board_svg": board_svg,
                "board_grid": board_to_grid(board),
                "move_history": move_history_str,
                "turn": "White" if board.turn == chess.WHITE else "Black",
                "top_moves": top_moves,
                "position_idx": position_idx,
                "original_moves": moves,
                "pgn": pgn_string,
            }

            return current_position_data

        except StopIteration:
            # Reset iterator if we reach the end
            dataset_iter = iter(dataset)
            continue


def format_chess_transcript(
    raw_transcript: str, board_fen: str, turn: str
) -> Dict[str, Optional[str]]:
    """
    Use GPT-4o to format a chess transcript

    Returns dict with 'reasoning_trace' and 'final_move'
    """
    if not openai_client:
        logger.warning("OpenAI client not initialized, returning raw transcript")
        return {"reasoning_trace": raw_transcript, "final_move": None}

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

Format the reasoning trace and extract the final move recommendation. Return a JSON object with keys "reasoning_trace" and "final_move"."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Using mini for cost efficiency
            messages=[
                {"role": "system", "content": system_prompt.format(turn=turn)},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        result = json.loads(response.choices[0].message.content)
        return {
            "reasoning_trace": result.get("reasoning_trace", raw_transcript),
            "final_move": result.get("final_move"),
        }

    except Exception as e:
        logger.error(f"Error formatting transcript: {e}")
        return {"reasoning_trace": raw_transcript, "final_move": None}


@app.route("/")
def index():
    """Serve the main page"""
    return render_template("index.html")


@app.route("/api/next_position", methods=["GET"])
def next_position():
    """Get the next chess position"""
    position_data = get_next_position()

    return jsonify(
        {
            "board_svg": position_data["board_svg"],
            "board_fen": position_data["board_fen"],
            "move_history": position_data["move_history"],
            "turn": position_data["turn"],
            "top_moves": position_data["top_moves"],
            "board_grid": position_data["board_grid"],
            "pgn": position_data["pgn"],
        }
    )


@app.route("/api/save_recording", methods=["POST"])
def save_recording():
    """Save audio recording and transcribe it"""
    try:
        # Get the audio file from the request
        audio_file = request.files["audio"]
        logger.info(
            f"Received audio file: {audio_file.filename}, size: {audio_file.content_length}"
        )

        # Save audio to temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = os.path.join(AUDIO_DIR, f"recording_{timestamp}.webm")
        audio_file.save(temp_path)
        logger.info(f"Saved audio to: {temp_path}")

        # Convert to WAV for transcription
        wav_path = temp_path.replace(".webm", ".wav")

        # Use ffmpeg to convert webm to wav
        import subprocess

        try:
            result = subprocess.run(
                ["ffmpeg", "-i", temp_path, "-ar", "16000", "-ac", "1", "-y", wav_path],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(f"Converted to WAV: {wav_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr}")
            return (
                jsonify(
                    {"success": False, "error": f"Audio conversion failed: {e.stderr}"}
                ),
                500,
            )

        # Check if WAV file exists and has content
        if not os.path.exists(wav_path):
            logger.error("WAV file was not created")
            return jsonify({"success": False, "error": "WAV file creation failed"}), 500

        wav_size = os.path.getsize(wav_path)
        logger.info(f"WAV file size: {wav_size} bytes")

        if wav_size == 0:
            logger.error("WAV file is empty")
            return (
                jsonify({"success": False, "error": "Converted audio file is empty"}),
                500,
            )

        # Transcribe with OpenAI GPT-4o mini
        try:
            with open(wav_path, "rb") as audio_file:
                logger.info("Sending to OpenAI for transcription...")
                transcript_response = openai_client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=audio_file,
                    language="en",  # Assuming English for chess analysis
                    temperature=0.0,  # Lower temperature for more accurate transcription
                )
            transcription = transcript_response.text.strip()
            logger.info(f"Transcription received: {transcription[:100]}...")
        except Exception as e:
            logger.error(f"Error with OpenAI transcription: {e}")
            return (
                jsonify({"success": False, "error": f"Transcription failed: {str(e)}"}),
                500,
            )

        # Format with GPT-4o if available
        # TODO: not needed anymore when using 4o mini
        #formatted_result = format_chess_transcript(
            #transcription,
            #current_position_data["board_fen"],
            #current_position_data["turn"],
        #)
        #print(f"Formatted result: {formatted_result}")

        # Save the dataset entry with all fields
        entry = {
            "analysis": transcription,  # Original whisper transcript
            "transcript_formatted": transcription,
            "final_move": "",
            "move_history": current_position_data["move_history"],
            "turn": current_position_data["turn"],
            "board_fen": current_position_data["board_fen"],
            "board_grid": current_position_data["board_grid"],
            "top_moves": current_position_data["top_moves"],
            "position_idx": current_position_data["position_idx"],
            "original_moves": current_position_data["original_moves"],
            "audio_file": f"recording_{timestamp}.webm",
            "timestamp": timestamp,
            "gpt_formatted": openai_client is not None,
        }

        # Append to dataset file
        dataset_file = os.path.join(OUTPUT_DIR, "manual_chess_dataset.jsonl")
        with open(dataset_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Clean up temporary WAV file
        os.remove(wav_path)

        # Keep track of the latest entry ID for deletion
        global last_entry_id
        last_entry_id = timestamp

        return jsonify(
            {
                "success": True,
                "transcription": transcription,
                "formatted_transcript": transcription,
                "final_move": "",
                "entry_id": timestamp,
            }
        )

    except Exception as e:
        logger.error(f"Error saving recording: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/update_analysis", methods=["POST"])
def update_analysis():
    """Update the analysis text for an existing entry"""
    try:
        data = request.json
        entry_id = data.get("entry_id")
        new_analysis = data.get("analysis")
        new_formatted = data.get("transcript_formatted")
        new_final_move = data.get("final_move")

        if not entry_id:
            return jsonify({"success": False, "error": "Missing entry_id"})

        dataset_file = os.path.join(OUTPUT_DIR, "manual_chess_dataset.jsonl")

        # Read all entries
        entries = []
        updated = False

        if os.path.exists(dataset_file):
            with open(dataset_file, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    # Find and update the matching entry
                    if entry.get("timestamp") == entry_id:
                        if new_analysis is not None:
                            entry["analysis"] = new_analysis
                        if new_formatted is not None:
                            entry["transcript_formatted"] = new_formatted
                        if new_final_move is not None:
                            entry["final_move"] = new_final_move
                        entry["edited"] = True
                        entry["edited_at"] = datetime.now().isoformat()
                        updated = True
                    entries.append(entry)

        if not updated:
            return jsonify({"success": False, "error": "Entry not found"})

        # Rewrite the file with updated entries
        with open(dataset_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        return jsonify({"success": True, "message": "Analysis updated successfully"})

    except Exception as e:
        logger.error(f"Error updating analysis: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/delete_last", methods=["POST"])
def delete_last_recording():
    """Delete or mark the last recording as invalid"""
    try:
        dataset_file = os.path.join(OUTPUT_DIR, "manual_chess_dataset.jsonl")

        # Read all entries
        entries = []
        if os.path.exists(dataset_file):
            with open(dataset_file, "r") as f:
                for line in f:
                    entries.append(json.loads(line))

        if not entries:
            return jsonify({"success": False, "error": "No entries to delete"})

        # Mark the last entry as invalid instead of deleting
        # This preserves the data in case you change your mind
        last_entry = entries[-1]
        last_entry["invalid"] = True
        last_entry["invalidated_at"] = datetime.now().isoformat()

        # Rewrite the file with the updated entry
        with open(dataset_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        return jsonify({"success": True, "message": "Last entry marked as invalid"})

    except Exception as e:
        logger.error(f"Error deleting last recording: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/dataset_stats", methods=["GET"])
def dataset_stats():
    """Get statistics about the created dataset"""
    dataset_file = os.path.join(OUTPUT_DIR, "manual_chess_dataset.jsonl")

    if not os.path.exists(dataset_file):
        return jsonify({"count": 0, "valid_count": 0})

    total_count = 0
    valid_count = 0

    with open(dataset_file, "r") as f:
        for line in f:
            total_count += 1
            entry = json.loads(line)
            if not entry.get("invalid", False):
                valid_count += 1

    return jsonify({"count": total_count, "valid_count": valid_count})


# Create the HTML template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Chess Dataset Creator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            display: flex;
            gap: 20px;
        }
        .board-section {
            flex: 1;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .info-section {
            flex: 1;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .controls {
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .recording {
            background-color: #f44336;
        }
        .recording:hover {
            background-color: #da190b;
        }
        .move-list {
            margin-top: 10px;
        }
        .move-item {
            padding: 5px;
            margin: 5px 0;
            border-radius: 4px;
        }
        .move-1 { background-color: #90EE90; }
        .move-2 { background-color: #FFFFE0; }
        .move-3 { background-color: #FFE4B5; }
        .transcription {
            margin-top: 20px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 4px;
            min-height: 100px;
        }
        #transcription-text, #formatted-text {
            width: 100%;
            min-height: 150px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-family: Arial, sans-serif;
            font-size: 14px;
            resize: vertical;
        }
        #final-move {
            width: 200px;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-family: monospace;
            font-size: 14px;
        }
        .field-label {
            font-weight: bold;
            margin-top: 15px;
            margin-bottom: 5px;
        }
        .save-edit-btn {
            background-color: #4CAF50;
            margin-top: 10px;
        }
        .save-edit-btn:hover {
            background-color: #45a049;
        }
        .edit-status {
            margin-left: 10px;
            color: #4CAF50;
            font-style: italic;
        }
        .stats {
            position: fixed;
            top: 10px;
            right: 10px;
            background: white;
            padding: 10px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .delete-btn {
            background-color: #ff6b6b;
        }
        .delete-btn:hover {
            background-color: #ff5252;
        }
        .pgn-section {
            margin-top: 20px;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            word-break: break-all;
        }
        .copy-btn {
            background-color: #2196F3;
            padding: 5px 10px;
            font-size: 14px;
            margin-top: 5px;
        }
        .copy-btn:hover {
            background-color: #0b7dda;
        }
    </style>
</head>
<body>
    <div class="stats">
        <strong>Total entries: <span id="dataset-count">0</span></strong><br>
        <strong>Valid entries: <span id="valid-count">0</span></strong>
    </div>
    
    <h1>Chess Dataset Creator</h1>
    
    <div class="container">
        <div class="board-section">
            <div id="board"></div>
            <div class="controls">
                <button id="record-btn" onclick="toggleRecording()">Start Recording</button>
                <button onclick="nextPosition()">Next Position</button>
                <button class="delete-btn" onclick="deleteLastRecording()">Delete Last Recording</button>
            </div>
        </div>
        
        <div class="info-section">
            <h3>Position Info</h3>
            <p><strong>Turn:</strong> <span id="turn"></span></p>
            <p><strong>Move History:</strong> <span id="move-history"></span></p>
            
            <h3>Top Stockfish Moves</h3>
            <div id="top-moves" class="move-list"></div>
            
            <h3>Your Analysis</h3>
            <div id="transcription" class="transcription">
                <div class="field-label">Raw Transcript (Whisper):</div>
                <textarea id="transcription-text" placeholder="Press 'Start Recording' and analyze the position..."></textarea>
                
                <div class="field-label">Formatted Transcript (GPT-4o cleaned):</div>
                <textarea id="formatted-text" placeholder="Formatted transcript will appear here..."></textarea>
                
                <div class="field-label">Final Move (UCI format, e.g., e2e4):</div>
                <input type="text" id="final-move" placeholder="e2e4">
                
                <button class="save-edit-btn" onclick="saveEdit()" style="display:none;">Save All Edits</button>
                <span id="edit-status" class="edit-status" style="display:none;"></span>
            </div>
            
            <h3>PGN (for Lichess import)</h3>
            <div class="pgn-section">
                <div id="pgn-text"></div>
                <button class="copy-btn" onclick="copyPGN()">Copy PGN</button>
            </div>
        </div>
    </div>
    
    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let currentPGN = '';
        let currentEntryId = null;
        
        // Initialize
        window.onload = function() {
            nextPosition();
            updateStats();
            
            // Add change listeners to all editable fields
            document.getElementById('transcription-text').addEventListener('input', showSaveButton);
            document.getElementById('formatted-text').addEventListener('input', showSaveButton);
            document.getElementById('final-move').addEventListener('input', showSaveButton);
        };
        
        function showSaveButton() {
            if (currentEntryId) {
                document.querySelector('.save-edit-btn').style.display = 'inline-block';
                document.getElementById('edit-status').style.display = 'none';
            }
        }
        
        async function nextPosition() {
            const response = await fetch('/api/next_position');
            const data = await response.json();
            
            // Update board
            document.getElementById('board').innerHTML = data.board_svg;
            
            // Update info
            document.getElementById('turn').textContent = data.turn;
            document.getElementById('move-history').textContent = data.move_history;
            
            // Update top moves
            const movesHtml = data.top_moves.map((move, i) => `
                <div class="move-item move-${i+1}">
                    ${i+1}. ${move.san} (${move.move}) - Eval: ${move.evaluation}
                </div>
            `).join('');
            document.getElementById('top-moves').innerHTML = movesHtml;
            
            // Update PGN
            currentPGN = data.pgn;
            document.getElementById('pgn-text').textContent = data.pgn;
            
            // Clear previous transcription and formatted fields
            document.getElementById('transcription-text').value = '';
            document.getElementById('formatted-text').value = '';
            document.getElementById('final-move').value = '';
            document.querySelector('.save-edit-btn').style.display = 'none';
            document.getElementById('edit-status').style.display = 'none';
            currentEntryId = null;
        }
        
        function copyPGN() {
            navigator.clipboard.writeText(currentPGN).then(() => {
                const btn = event.target;
                const originalText = btn.textContent;
                btn.textContent = 'Copied!';
                setTimeout(() => {
                    btn.textContent = originalText;
                }, 2000);
            });
        }
        
        async function deleteLastRecording() {
            if (!confirm('Are you sure you want to delete the last recording?')) {
                return;
            }
            
            try {
                const response = await fetch('/api/delete_last', {
                    method: 'POST'
                });
                
                const data = await response.json();
                if (data.success) {
                    alert('Last recording marked as invalid');
                    updateStats();
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error deleting recording');
                console.error(error);
            }
        }
        
        async function toggleRecording() {
            if (!isRecording) {
                // Start recording
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = event => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };
                
                mediaRecorder.onstop = async () => {
                    console.log('Recording stopped, chunks:', audioChunks.length);
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    console.log('Created blob, size:', audioBlob.size);
                    if (audioBlob.size > 0) {
                        await uploadRecording(audioBlob);
                    } else {
                        alert('No audio data recorded. Please try again.');
                    }
                };
                
                mediaRecorder.start();
                isRecording = true;
                document.getElementById('record-btn').textContent = 'Stop Recording';
                document.getElementById('record-btn').classList.add('recording');
            } else {
                // Stop recording
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                document.getElementById('record-btn').textContent = 'Start Recording';
                document.getElementById('record-btn').classList.remove('recording');
            }
        }
        
        async function uploadRecording(audioBlob) {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.webm');
            
            document.getElementById('transcription-text').value = 'Transcribing with GPT-4o mini...';
            document.getElementById('formatted-text').value = 'Waiting for transcription...';
            document.getElementById('final-move').value = '';
            
            console.log('Uploading audio blob:', audioBlob.size, 'bytes');
            
            try {
                const response = await fetch('/api/save_recording', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                console.log('Server response:', data);
                
                if (data.success) {
                    document.getElementById('transcription-text').value = data.transcription;
                    document.getElementById('formatted-text').value = data.formatted_transcript || data.transcription;
                    document.getElementById('final-move').value = data.final_move || '';
                    currentEntryId = data.entry_id;
                    updateStats();
                } else {
                    document.getElementById('transcription-text').value = 'Error: ' + data.error;
                    console.error('Server error:', data.error);
                }
            } catch (error) {
                document.getElementById('transcription-text').value = 'Error uploading recording';
                console.error('Upload error:', error);
            }
        }
        
        async function saveEdit() {
            if (!currentEntryId) {
                alert('No entry to edit');
                return;
            }
            
            const newAnalysis = document.getElementById('transcription-text').value;
            const newFormatted = document.getElementById('formatted-text').value;
            const newFinalMove = document.getElementById('final-move').value;
            
            try {
                const response = await fetch('/api/update_analysis', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        entry_id: currentEntryId,
                        analysis: newAnalysis,
                        transcript_formatted: newFormatted,
                        final_move: newFinalMove || null
                    })
                });
                
                const data = await response.json();
                if (data.success) {
                    document.querySelector('.save-edit-btn').style.display = 'none';
                    document.getElementById('edit-status').style.display = 'inline';
                    document.getElementById('edit-status').textContent = '✓ Saved';
                    
                    // Hide the status after 2 seconds
                    setTimeout(() => {
                        document.getElementById('edit-status').style.display = 'none';
                    }, 2000);
                } else {
                    alert('Error saving edit: ' + data.error);
                }
            } catch (error) {
                alert('Error saving edit');
                console.error(error);
            }
        }
        
        async function updateStats() {
            const response = await fetch('/api/dataset_stats');
            const data = await response.json();
            document.getElementById('dataset-count').textContent = data.count;
            document.getElementById('valid-count').textContent = data.valid_count;
        }
    </script>
</body>
</html>
"""

# Create templates directory and save the HTML
os.makedirs("templates", exist_ok=True)
with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(html_template)

if __name__ == "__main__":
    try:
        logger.info("Initializing resources...")
        init_resources()
        logger.info("Starting Flask server on http://localhost:5000")
        app.run(debug=True, port=5000)
    finally:
        if engine:
            engine.quit()
