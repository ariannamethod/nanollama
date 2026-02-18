#!/usr/bin/env python3
"""Compare base vs personality model generation."""

import torch
import sentencepiece as spm
from nanollama.llama import Llama, get_config_for_depth


def load_model(ckpt_path, device):
    config = get_config_for_depth(12)
    config.vocab_size = 32000
    model = Llama(config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def generate(model, sp, prompt, device, temp=0.8, max_tokens=150):
    ids = torch.tensor([sp.encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(max_tokens):
            logits = model(ids)[:, -1, :]
            probs = torch.softmax(logits / temp, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
    return sp.decode(ids[0].tolist())


def main():
    device = torch.device("cuda")
    sp = spm.SentencePieceProcessor(
        model_file="/home/ubuntu/.cache/nanollama/tokenizer/tokenizer.model"
    )
    print(f"Tokenizer: {sp.get_piece_size()} pieces")

    print("Loading BASE model...")
    base = load_model(
        "/home/ubuntu/.cache/nanollama/checkpoints/micro-d12-base/checkpoint_step3000.pt",
        device,
    )
    print("Loading PERSONALITY model...")
    pers = load_model(
        "/home/ubuntu/.cache/nanollama/checkpoints/micro-d12/checkpoint_step3000.pt",
        device,
    )

    prompts = [
        "What is the meaning of life",
        "Tell me about love and loss",
        "The universe is",
        "I think therefore",
        "Once upon a time there was",
    ]

    for prompt in prompts:
        print()
        print("=" * 60)
        print(f"PROMPT: {prompt}")
        print("=" * 60)

        torch.manual_seed(42)
        out_base = generate(base, sp, prompt, device)
        print()
        print("--- BASE ---")
        print(out_base[:400])

        torch.manual_seed(42)
        out_pers = generate(pers, sp, prompt, device)
        print()
        print("--- PERSONALITY (20% Yent) ---")
        print(out_pers[:400])


if __name__ == "__main__":
    main()
