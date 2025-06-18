#!/usr/bin/env python
# coding: utf-8

import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_prompt_completion(input_file, output_file):
    """Convert HF format dataset to prompt-completion format"""

    converted_count = 0

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())

                # Extract messages
                messages = data.get("messages", [])

                # Find system and user messages for prompt
                prompt_messages = []
                completion_content = None

                for msg in messages:
                    if msg["role"] in ["system", "user"]:
                        prompt_messages.append(msg)
                    elif msg["role"] == "assistant":
                        # In your format, there should only be one assistant message
                        completion_content = msg["content"]

                if not prompt_messages or not completion_content:
                    logger.warning(
                        f"Skipping line {line_num}: Missing required messages"
                    )
                    continue

                # Create prompt-completion format
                # Using conversational format since you have system + user messages
                prompt_completion = {
                    "prompt": prompt_messages,
                    "completion": [
                        {"role": "assistant", "content": completion_content}
                    ],
                }

                f_out.write(json.dumps(prompt_completion) + "\n")
                converted_count += 1

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing line {line_num}: {e}")
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")

    logger.info(f"Successfully converted {converted_count} examples")
    return converted_count


def convert_to_simple_prompt_completion(input_file, output_file):
    """Convert to simple prompt-completion format (just assistant content as completion)"""

    converted_count = 0

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())

                # Extract messages
                messages = data.get("messages", [])

                # Build the prompt string (everything except assistant response)
                prompt_parts = []
                completion = None

                for msg in messages:
                    if msg["role"] == "system":
                        prompt_parts.append(f"System: {msg['content']}")
                    elif msg["role"] == "user":
                        prompt_parts.append(f"User: {msg['content']}")
                    elif msg["role"] == "assistant":
                        completion = msg["content"]

                if not prompt_parts or not completion:
                    logger.warning(
                        f"Skipping line {line_num}: Missing required messages"
                    )
                    continue

                # Join prompt parts
                prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"

                # Create simple prompt-completion format
                prompt_completion = {
                    "prompt": prompt,
                    "completion": " "
                    + completion,  # Space prefix is common for completions
                }

                f_out.write(json.dumps(prompt_completion) + "\n")
                converted_count += 1

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing line {line_num}: {e}")
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")

    logger.info(f"Successfully converted {converted_count} examples")
    return converted_count


def convert_with_chat_template(
    input_file, output_file, tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct"
):
    """Convert using the actual chat template from the tokenizer"""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    converted_count = 0

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())
                messages = data.get("messages", [])

                # Split messages into prompt (system + user) and completion (assistant)
                prompt_messages = []
                completion = None

                for msg in messages:
                    if msg["role"] in ["system", "user"]:
                        prompt_messages.append(msg)
                    elif msg["role"] == "assistant":
                        completion = msg["content"]

                if not prompt_messages or not completion:
                    logger.warning(
                        f"Skipping line {line_num}: Missing required messages"
                    )
                    continue

                # Apply chat template to prompt messages and add generation prompt
                prompt = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )

                # For the completion, we want just the assistant's content
                # but we might want to add the thinking prefix
                completion_with_prefix = (
                    "<think>" + completion[7:]
                )  # Skip existing <think> if present

                prompt_completion = {
                    "prompt": prompt + "<think>",  # Add thinking prefix to prompt
                    "completion": completion[
                        7:
                    ],  # Remove <think> from completion since it's in prompt
                }

                f_out.write(json.dumps(prompt_completion) + "\n")
                converted_count += 1

            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")

    logger.info(f"Successfully converted {converted_count} examples")
    return converted_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert dataset to prompt-completion format"
    )
    parser.add_argument("input_file", help="Input JSONL file")
    parser.add_argument("output_file", help="Output JSONL file")
    parser.add_argument(
        "--format",
        choices=["conversational", "simple", "chat_template"],
        default="conversational",
        help="Output format type",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Tokenizer to use for chat_template format",
    )

    args = parser.parse_args()

    logger.info(
        f"Converting {args.input_file} to {args.output_file} using {args.format} format"
    )

    if args.format == "conversational":
        convert_to_prompt_completion(args.input_file, args.output_file)
    elif args.format == "simple":
        convert_to_simple_prompt_completion(args.input_file, args.output_file)
    elif args.format == "chat_template":
        convert_with_chat_template(args.input_file, args.output_file, args.tokenizer)

    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
