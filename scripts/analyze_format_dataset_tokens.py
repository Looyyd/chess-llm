#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import logging
from tqdm import tqdm
import os
from datetime import datetime

# Try to import matplotlib, but don't fail if not available
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not available, plots will be skipped")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_token_lengths(dataset_path, model_path, max_length=4096):
    """Analyze token lengths in the format alignment dataset"""

    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)

    # Load the tokenizer from the model being used in training
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the dataset
    logger.info(f"Loading dataset from {dataset_path}")
    if dataset_path.startswith("Looyyd/"):
        # Load from Hugging Face
        dataset = load_dataset(
            dataset_path,
            data_files={"train": "train.jsonl"},
            split="train",
        )
    else:
        # Load from local file
        dataset = load_dataset("json", data_files=dataset_path, split="train")

    logger.info(f"Dataset size: {len(dataset)} examples")

    # Analyze token lengths
    token_lengths = []
    input_lengths = []
    completion_lengths = []
    over_limit_count = 0
    over_limit_examples = []

    logger.info("Tokenizing examples...")
    for idx, example in enumerate(tqdm(dataset, desc="Processing examples")):
        # Handle different dataset formats
        if "messages" in example:
            messages = example["messages"]
            
            # Split messages into input (system + user) and completion (assistant)
            input_messages = []
            completion_messages = []
            
            for msg in messages:
                if msg["role"] in ["system", "user"]:
                    input_messages.append(msg)
                elif msg["role"] == "assistant":
                    completion_messages.append(msg)
            
            # Apply chat template to get the full text
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False)
                
                # Tokenize input part
                if input_messages:
                    input_text = tokenizer.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True)
                    input_tokens = tokenizer(input_text, return_tensors=None)
                    input_token_count = len(input_tokens["input_ids"])
                    input_lengths.append(input_token_count)
                
                # Calculate completion tokens (total - input)
                total_tokens = tokenizer(text, return_tensors=None)
                total_token_count = len(total_tokens["input_ids"])
                completion_token_count = total_token_count - input_token_count
                completion_lengths.append(completion_token_count)
                
            except Exception as e:
                logger.error(f"Error applying chat template at index {idx}: {e}")
                continue

        elif "prompt" in example and "completion" in example:
            # Handle the prompt-completion format
            # Concatenate prompt and completion messages
            all_messages = []
            prompt_messages = []
            completion_messages = []

            # Add prompt messages
            if isinstance(example["prompt"], list):
                prompt_messages.extend(example["prompt"])
                all_messages.extend(example["prompt"])
            else:
                logger.warning(f"Unexpected prompt format at index {idx}")
                continue

            # Add completion messages
            if isinstance(example["completion"], list):
                completion_messages.extend(example["completion"])
                all_messages.extend(example["completion"])
            else:
                logger.warning(f"Unexpected completion format at index {idx}")
                continue

            # Apply chat template
            try:
                # Tokenize full text
                text = tokenizer.apply_chat_template(all_messages, tokenize=False)
                
                # Tokenize input part
                input_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                input_tokens = tokenizer(input_text, return_tensors=None)
                input_token_count = len(input_tokens["input_ids"])
                input_lengths.append(input_token_count)
                
                # Calculate completion tokens
                total_tokens = tokenizer(text, return_tensors=None)
                total_token_count = len(total_tokens["input_ids"])
                completion_token_count = total_token_count - input_token_count
                completion_lengths.append(completion_token_count)
                
            except Exception as e:
                logger.error(f"Error applying chat template at index {idx}: {e}")
                continue
        else:
            logger.warning(f"Unknown format at index {idx}: {example.keys()}")
            continue

        # Store total token count
        try:
            token_count = total_token_count
            token_lengths.append(token_count)

            # Check if over limit
            if token_count > max_length:
                over_limit_count += 1

                # Extract move history from the text if available
                move_history = "N/A"
                if "Move history" in text:
                    try:
                        # Extract move history from the user prompt
                        start = text.find("Move history (UCI format): ")
                        if start != -1:
                            start += len("Move history (UCI format): ")
                            end = text.find("\n", start)
                            if end != -1:
                                move_history = text[start:end].strip()
                    except:
                        pass

                over_limit_examples.append(
                    {
                        "index": idx,
                        "token_count": token_count,
                        "input_token_count": input_token_count,
                        "completion_token_count": completion_token_count,
                        "move_history": move_history,
                        "text_preview": text[:200] + "..." if len(text) > 200 else text,
                    }
                )
        except Exception as e:
            logger.error(f"Error tokenizing example {idx}: {e}")
            continue

    # Calculate statistics
    token_lengths = np.array(token_lengths)
    input_lengths = np.array(input_lengths)
    completion_lengths = np.array(completion_lengths)

    def print_stats(name, lengths):
        logger.info(f"\n{name} STATISTICS:")
        logger.info("-" * 40)
        logger.info(f"Maximum: {np.max(lengths)}")
        logger.info(f"Minimum: {np.min(lengths)}")
        logger.info(f"Mean: {np.mean(lengths):.2f}")
        logger.info(f"Median: {np.median(lengths):.2f}")
        logger.info(f"Standard deviation: {np.std(lengths):.2f}")
        logger.info(f"Percentiles:")
        for p in [50, 75, 90, 95, 99, 99.9]:
            logger.info(f"  {p}th percentile: {np.percentile(lengths, p):.0f} tokens")

    logger.info("\n" + "=" * 60)
    logger.info("TOKEN LENGTH STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total examples analyzed: {len(token_lengths)}")
    
    print_stats("TOTAL TOKEN", token_lengths)
    print_stats("INPUT TOKEN", input_lengths)
    print_stats("COMPLETION TOKEN", completion_lengths)

    logger.info(
        f"\nExamples over {max_length} tokens: {over_limit_count} ({over_limit_count/len(token_lengths)*100:.2f}%)"
    )

    # Show examples that are over the limit
    if over_limit_examples:
        logger.info(f"\nFirst 5 examples over {max_length} tokens:")
        for i, example in enumerate(over_limit_examples[:5]):
            logger.info(f"\n  Example {i+1}:")
            logger.info(f"    Index: {example['index']}")
            logger.info(f"    Total tokens: {example['token_count']} (Input: {example['input_token_count']}, Completion: {example['completion_token_count']})")
            logger.info(f"    Move history: {example['move_history']}")

    # Save detailed report to file
    report_filename = os.path.join(
        "./data", f"token_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    
    def write_stats(f, name, lengths):
        f.write(f"\n{name} STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Maximum: {np.max(lengths)}\n")
        f.write(f"Minimum: {np.min(lengths)}\n")
        f.write(f"Mean: {np.mean(lengths):.2f}\n")
        f.write(f"Median: {np.median(lengths):.2f}\n")
        f.write(f"Standard deviation: {np.std(lengths):.2f}\n")
        f.write("Percentiles:\n")
        for p in [50, 75, 90, 95, 99, 99.9]:
            f.write(f"  {p}th percentile: {np.percentile(lengths, p):.0f} tokens\n")
    
    with open(report_filename, "w") as f:
        f.write("TOKEN LENGTH ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Model/Tokenizer: {model_path}\n")
        f.write(f"Max length setting: {max_length}\n")
        f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        f.write(f"Total examples analyzed: {len(token_lengths)}\n")
        
        write_stats(f, "TOTAL TOKEN", token_lengths)
        write_stats(f, "INPUT TOKEN", input_lengths)
        write_stats(f, "COMPLETION TOKEN", completion_lengths)
        
        f.write(
            f"\nExamples over {max_length} tokens: {over_limit_count} ({over_limit_count/len(token_lengths)*100:.2f}%)\n"
        )

        if over_limit_examples:
            f.write(
                f"\nAll {len(over_limit_examples)} examples over {max_length} tokens:\n"
            )
            for i, example in enumerate(over_limit_examples):
                f.write(f"\n  Example {i+1}:\n")
                f.write(f"    Index: {example['index']}\n")
                f.write(f"    Total tokens: {example['token_count']} (Input: {example['input_token_count']}, Completion: {example['completion_token_count']})\n")
                f.write(f"    Move history: {example['move_history']}\n")

    logger.info(f"\nDetailed report saved to '{report_filename}'")

    # Create plots if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        # Create total length histogram
        plt.figure(figsize=(10, 6))
        plt.hist(token_lengths, bins=50, alpha=0.7, edgecolor="black")
        plt.axvline(
            max_length,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Max length ({max_length})",
        )
        plt.axvline(
            np.mean(token_lengths),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Mean ({np.mean(token_lengths):.0f})",
        )
        plt.axvline(
            np.median(token_lengths),
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Median ({np.median(token_lengths):.0f})",
        )
        plt.xlabel("Token Length")
        plt.ylabel("Frequency")
        plt.title("Distribution of Total Token Lengths in Format Alignment Dataset")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("./data/format_dataset_token_distribution.png", dpi=300)
        logger.info("\nHistogram saved as './data/format_dataset_token_distribution.png'")

        # Create cumulative distribution plot
        plt.figure(figsize=(10, 6))
        sorted_lengths = np.sort(token_lengths)
        cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
        plt.plot(sorted_lengths, cumulative, linewidth=2)
        plt.axvline(
            max_length,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Max length ({max_length})",
        )
        plt.axhline(95, color="gray", linestyle=":", alpha=0.7, label="95%")
        plt.axhline(99, color="gray", linestyle=":", alpha=0.7, label="99%")
        plt.xlabel("Token Length")
        plt.ylabel("Cumulative Percentage (%)")
        plt.title("Cumulative Distribution of Total Token Lengths")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("./data/format_dataset_token_cumulative.png", dpi=300)
        logger.info(
            "Cumulative distribution plot saved as './data/format_dataset_token_cumulative.png'"
        )
        
        # Create comparison plot for input vs completion lengths
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(input_lengths, bins=50, alpha=0.7, edgecolor="black", label="Input", color="blue")
        plt.hist(completion_lengths, bins=50, alpha=0.7, edgecolor="black", label="Completion", color="green")
        plt.xlabel("Token Length")
        plt.ylabel("Frequency")
        plt.title("Distribution of Input vs Completion Token Lengths")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        data = [input_lengths, completion_lengths, token_lengths]
        labels = ["Input", "Completion", "Total"]
        plt.boxplot(data, labels=labels)
        plt.ylabel("Token Length")
        plt.title("Token Length Comparison")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("./data/format_dataset_token_comparison.png", dpi=300)
        logger.info("Comparison plot saved as './data/format_dataset_token_comparison.png'")
    else:
        logger.info("\nPlots skipped (matplotlib not available)")

    return token_lengths, over_limit_examples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze token lengths in format alignment dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Looyyd/chess-format-alignment",
        help="Dataset path (HuggingFace or local file)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./chess_sft_qwen_hf/checkpoint-5000/",
        help="Model path for tokenizer",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum token length to check against",
    )
    parser.add_argument(
        "--local-file",
        type=str,
        help="Local dataset file to analyze (overrides --dataset)",
    )

    args = parser.parse_args()

    # Use local file if provided, otherwise use the dataset argument
    dataset_path = args.local_file if args.local_file else args.dataset

    analyze_token_lengths(dataset_path, args.model, args.max_length)
