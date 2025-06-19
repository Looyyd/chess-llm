#!/usr/bin/env python
# coding: utf-8

import sys
import os
# Add parent directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from termcolor import colored
from utils.prompt_utils import get_chess_system_prompt, create_chess_user_prompt, format_chess_messages


def get_color_for_index(idx):
    """Get a color for a given token index"""
    colors = ["red", "green", "yellow", "blue", "magenta", "cyan"]
    return colors[idx % len(colors)]


def analyze_prefill_tokenization(tokenizer, prompt, prefill_text):
    """Analyze how prefill text is tokenized when added to a prompt"""
    
    print("=" * 80)
    print("TOKENIZATION ANALYSIS FOR PREFILL")
    print("=" * 80)
    
    # Tokenize the original prompt
    print("\n1. ORIGINAL PROMPT TOKENIZATION:")
    print("-" * 40)
    prompt_tokens = tokenizer.tokenize(prompt)
    prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
    print(f"Prompt ends with: {prompt_tokens[-5:]}")
    print(f"Last 5 token IDs: {prompt_ids[-5:]}")
    
    # Tokenize the prefill text alone
    print(f"\n2. PREFILL TEXT ALONE: {repr(prefill_text)}")
    print("-" * 40)
    prefill_tokens = tokenizer.tokenize(prefill_text, add_special_tokens=False)
    prefill_ids = tokenizer.convert_tokens_to_ids(prefill_tokens)
    print(f"Prefill tokens: {prefill_tokens}")
    print(f"Prefill token IDs: {prefill_ids}")
    
    # Tokenize the combined text
    combined_text = prompt + prefill_text
    print(f"\n3. COMBINED TEXT TOKENIZATION:")
    print("-" * 40)
    combined_tokens = tokenizer.tokenize(combined_text)
    combined_ids = tokenizer.convert_tokens_to_ids(combined_tokens)
    
    # Find where the difference starts
    for i, (orig, comb) in enumerate(zip(prompt_tokens, combined_tokens)):
        if orig != comb:
            print(f"Tokenization differs starting at position {i}")
            print(f"Original: {prompt_tokens[max(0, i-3):i+3]}")
            print(f"Combined: {combined_tokens[max(0, i-3):i+3]}")
            break
    
    print(f"Combined ends with: {combined_tokens[-10:]}")
    print(f"Last 10 token IDs: {combined_ids[-10:]}")
    
    # Check if we can decode the last few tokens to see what's happening
    print(f"\n4. DECODING ANALYSIS:")
    print("-" * 40)
    
    # Try to decode the last few tokens of the combined text
    last_tokens = combined_ids[-5:]
    for i, token_id in enumerate(last_tokens):
        token = tokenizer.decode([token_id])
        print(f"Token {len(combined_ids)-5+i}: ID={token_id}, decoded={repr(token)}")
    
    # Try the prefill approach with proper tokenization
    print(f"\n5. PROPER PREFILL TOKENIZATION:")
    print("-" * 40)
    
    # Tokenize prompt and prefill separately, then concatenate
    prompt_encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prefill_encoded = tokenizer(prefill_text, return_tensors="pt", add_special_tokens=False)
    
    print(f"Prompt token IDs shape: {prompt_encoded.input_ids.shape}")
    print(f"Prefill token IDs shape: {prefill_encoded.input_ids.shape}")
    
    # Concatenate
    combined_ids_proper = torch.cat([prompt_encoded.input_ids, prefill_encoded.input_ids], dim=1)
    print(f"Combined shape: {combined_ids_proper.shape}")
    
    # Decode to see what we get
    decoded_combined = tokenizer.decode(combined_ids_proper[0], skip_special_tokens=True)
    print(f"Properly combined text ends with: {repr(decoded_combined[-50:])}")
    
    return combined_ids_proper


def test_think_tag_generation(tokenizer, model_path="Qwen/Qwen2.5-7B-Instruct"):
    """Test different approaches to generating with <think> prefill"""
    
    print("\n" + "=" * 80)
    print("TESTING <think> TAG GENERATION APPROACHES")
    print("=" * 80)
    
    # Create a simple test prompt
    system_prompt = get_chess_system_prompt()
    user_prompt = """Current game position:

Move history (UCI format): e2e4 e7e5
Turn: White

Current board state:
  a b c d e f g h
  ----------------
8| r n b q k b n r |8
7| p p p p . p p p |7
6| . . . . . . . . |6
5| . . . . p . . . |5
4| . . . . P . . . |4
3| . . . . . . . . |3
2| P P P P . P P P |2
1| R N B Q K B N R |1
  ----------------
  a b c d e f g h

What is the best move? Analyze the position and provide your answer."""

    messages = format_chess_messages(system_prompt, user_prompt)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("Testing different prefill approaches:\n")
    
    # Approach 1: Simple string concatenation
    print("1. Simple string concatenation:")
    approach1 = prompt + "<think>"
    tokens1 = tokenizer.tokenize(approach1)
    print(f"   Last 5 tokens: {tokens1[-5:]}")
    
    # Approach 2: Tokenize separately and concatenate
    print("2. Separate tokenization and concatenation:")
    prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    think_tokens = tokenizer("<think>", return_tensors="pt", add_special_tokens=False)
    combined_tokens = torch.cat([prompt_tokens.input_ids, think_tokens.input_ids], dim=1)
    decoded = tokenizer.decode(combined_tokens[0])
    decoded_tokens = tokenizer.tokenize(decoded)
    print(f"   Last 5 tokens: {decoded_tokens[-5:]}")
    
    # Approach 3: Check if there's a special token for <think>
    print("3. Check if <think> is a special token:")
    vocab = tokenizer.get_vocab()
    think_variations = ["<think>", "<think", "think>", "think"]
    for variation in think_variations:
        if variation in vocab:
            print(f"   Found '{variation}' in vocab with ID: {vocab[variation]}")
        else:
            print(f"   '{variation}' not found in vocab")
    
    # Approach 4: Show what happens when we try to generate
    if os.path.exists(model_path) or model_path.startswith("Qwen/"):
        print("\n4. Testing actual generation (first few tokens only):")
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
            
            # Test string concatenation approach
            test_prompt = prompt + "<think>"
            inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,  # Just a few tokens to see what happens
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
            print(f"   Generated: {repr(generated_part)}")
            
        except Exception as e:
            print(f"   Could not load model: {e}")


def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading tokenizer: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Could not load tokenizer: {e}")
        print("Using local model path instead...")
        tokenizer = AutoTokenizer.from_pretrained("./chess_sft_qwen_hf/checkpoint-5000/")
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"EOS token: {repr(tokenizer.eos_token)} (id: {tokenizer.eos_token_id})")
    
    # Create a test prompt
    system_prompt = get_chess_system_prompt()
    user_prompt = create_chess_user_prompt(
        """  a b c d e f g h
  ----------------
8| r n b q k b n r |8
7| p p p p . p p p |7
6| . . . . . . . . |6
5| . . . . p . . . |5
4| . . . . P . . . |4
3| . . . . . . . . |3
2| P P P P . P P P |2
1| R N B Q K B N R |1
  ----------------
  a b c d e f g h""",
        "e2e4 e7e5",
        "White"
    )
    
    messages = format_chess_messages(system_prompt, user_prompt)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Analyze the prefill tokenization issue
    analyze_prefill_tokenization(tokenizer, prompt, "<think>")
    
    # Test different generation approaches
    test_think_tag_generation(tokenizer, model_name)


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