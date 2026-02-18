"""
CLI interface for nanollama chat.

Usage:
    python -m scripts.chat_cli --model-tag=chat
"""

import argparse
import sys

import torch

from nanollama.common import compute_init, autodetect_device_type, print0
from nanollama.checkpoint_manager import load_model
from nanollama.engine import Engine


def parse_args():
    parser = argparse.ArgumentParser(description="nanollama chat CLI")
    parser.add_argument("--model-tag", type=str, default="chat", help="Model to load")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize
    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)
    
    # Load model
    print0(f"Loading model: {args.model_tag}")
    try:
        model, tokenizer, meta = load_model(args.model_tag, device, phase="eval")
        engine = Engine(model, tokenizer)
        print0("Model loaded!\n")
    except Exception as e:
        print0(f"Error loading model: {e}")
        sys.exit(1)
    
    # Print header
    print("=" * 60)
    print("  🦙 nanollama Chat")
    print("  Type 'quit' or 'exit' to end the conversation")
    print("  Type 'clear' to clear conversation history")
    print("=" * 60)
    print()
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            conversation_history = []
            print("Conversation cleared.\n")
            continue
        
        # Add user message
        conversation_history.append({"role": "user", "content": user_input})
        
        # Prepare prompt
        conversation = {"messages": conversation_history}
        prompt_ids, _ = tokenizer.render_conversation(conversation)
        
        # Add assistant start token
        assistant_start = tokenizer.encode_special("<|assistant_start|>")
        if assistant_start:
            prompt_ids.append(assistant_start)
        
        # Generate response
        print("nanollama: ", end="", flush=True)
        
        generated_tokens = []
        for token_column, _ in engine.generate(
            prompt_ids,
            num_samples=1,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        ):
            token = token_column[0]
            
            # Check for end tokens
            if token == tokenizer.encode_special("<|assistant_end|>"):
                break
            if token == tokenizer.encode_special("<|eot_id|>"):
                break
            if token == tokenizer.get_bos_token_id():
                break
            
            generated_tokens.append(token)
            
            # Stream output
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
        
        print("\n")
        
        # Add assistant response to history
        response = tokenizer.decode(generated_tokens)
        conversation_history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
