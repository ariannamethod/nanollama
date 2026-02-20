#!/usr/bin/env python
"""
Train a multilingual SentencePiece BPE tokenizer for nanollama.

Three tiers of progressive multilingual support:
  Tier 1 (small, 48K):   EN, RU, FR, DE, ES — Latin + Cyrillic
  Tier 2 (medium, 64K):  Tier 1 + AR, HI, TR, PT, UK — new scripts
  Tier 3 (large, 96K+):  Tier 2 + ZH, JA, KO — CJK

Usage:
    python -m scripts.train_tokenizer --tier 1
    python -m scripts.train_tokenizer --tier 2 --vocab-size 64000
    python -m scripts.train_tokenizer --languages en,ru,fr --vocab-size 48000
    python -m scripts.train_tokenizer --tier 1 --samples-per-lang 200000

Data source: CulturaX (uonlp/CulturaX) on HuggingFace.
Output: tokenizer.model + special_tokens.txt in output directory.
"""

import argparse
import os
import sys
import tempfile
import time

# --- Language tiers ---

TIER_1 = {
    "en": "English",
    "ru": "Russian",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}

TIER_2 = {
    **TIER_1,
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "pt": "Portuguese",
    "uk": "Ukrainian",
}

TIER_3 = {
    **TIER_2,
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}

TIERS = {1: TIER_1, 2: TIER_2, 3: TIER_3}

DEFAULT_VOCAB = {1: 48000, 2: 64000, 3: 96000}

# Same special tokens as nanollama/tokenizer.py
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]


def download_culturax(languages: dict, samples_per_lang: int, output_file: str):
    """
    Download balanced text samples from CulturaX for each language.
    Writes all text to a single file for SentencePiece training.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library required. Install with: pip install datasets")
        sys.exit(1)

    print(f"\nDownloading CulturaX samples for {len(languages)} languages...")
    print(f"  Samples per language: {samples_per_lang:,}")
    print(f"  Total target: {samples_per_lang * len(languages):,} samples\n")

    total_written = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for lang_code, lang_name in languages.items():
            print(f"  [{lang_code}] {lang_name}...", end=" ", flush=True)
            t0 = time.time()

            try:
                ds = load_dataset(
                    "uonlp/CulturaX",
                    lang_code,
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )

                count = 0
                for sample in ds:
                    text = sample.get("text", "")
                    # Skip very short or very long texts
                    if len(text) < 50 or len(text) > 10000:
                        continue
                    # Write one line per sample (SentencePiece expects this)
                    # Replace newlines within text to keep one-sample-per-line
                    clean = text.replace("\n", " ").strip()
                    f.write(clean + "\n")
                    count += 1
                    if count >= samples_per_lang:
                        break

                dt = time.time() - t0
                print(f"{count:,} samples in {dt:.1f}s")
                total_written += count

            except Exception as e:
                print(f"FAILED: {e}")
                continue

    print(f"\nTotal: {total_written:,} samples written to {output_file}")
    return total_written


def download_fineweb_english(samples: int, output_file: str):
    """
    Download English samples from FineWeb-Edu (higher quality than CulturaX English).
    Appends to existing file.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library required. Install with: pip install datasets")
        sys.exit(1)

    print(f"\n  [en] FineWeb-Edu (high quality)...", end=" ", flush=True)
    t0 = time.time()

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    count = 0
    with open(output_file, "a", encoding="utf-8") as f:
        for sample in ds:
            text = sample.get("text", "")
            if len(text) < 50 or len(text) > 10000:
                continue
            clean = text.replace("\n", " ").strip()
            f.write(clean + "\n")
            count += 1
            if count >= samples:
                break

    dt = time.time() - t0
    print(f"{count:,} samples in {dt:.1f}s")
    return count


def train_sentencepiece(
    input_file: str,
    output_dir: str,
    vocab_size: int,
    num_languages: int,
):
    """Train SentencePiece BPE model on the prepared text file."""
    import sentencepiece as spm

    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, "tokenizer")

    # Reserve space for special tokens
    sp_vocab_size = vocab_size - len(SPECIAL_TOKENS)

    print(f"\nTraining SentencePiece BPE tokenizer...")
    print(f"  Vocab size: {sp_vocab_size:,} + {len(SPECIAL_TOKENS)} special = {vocab_size:,}")
    print(f"  Languages: {num_languages}")

    # Character coverage: higher for more diverse scripts
    # 0.9995 for Latin-only, 0.9999 for multilingual with CJK
    char_coverage = 0.9999 if num_languages > 5 else 0.9998

    t0 = time.time()
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=sp_vocab_size,
        model_type="bpe",
        character_coverage=char_coverage,
        num_threads=os.cpu_count(),
        split_digits=True,
        byte_fallback=True,
        # Multilingual-specific settings
        split_by_unicode_script=True,  # Don't merge across writing systems
        normalization_rule_name="identity",  # No normalization — preserve original text
        # Training efficiency
        input_sentence_size=5_000_000,  # Cap training sentences for speed
        shuffle_input_sentence=True,
        # Rare piece handling
        max_sentencepiece_length=16,
        seed_sentencepiece_size=1_000_000,
    )
    dt = time.time() - t0
    print(f"  Trained in {dt:.1f}s")

    # Load and verify
    sp = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
    print(f"  Base vocab: {sp.get_piece_size():,} pieces")

    # Save special tokens mapping
    base_vocab = sp.get_piece_size()
    special_path = os.path.join(output_dir, "special_tokens.txt")
    with open(special_path, "w") as f:
        for i, token in enumerate(SPECIAL_TOKENS):
            f.write(f"{token}\t{base_vocab + i}\n")
    print(f"  Special tokens: {len(SPECIAL_TOKENS)} (IDs {base_vocab}–{base_vocab + len(SPECIAL_TOKENS) - 1})")

    return sp


def verify_tokenizer(model_path: str, languages: dict):
    """Verify tokenizer works on sample text from each language."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=model_path)

    # Test strings per language
    test_strings = {
        "en": "The quick brown fox jumps over the lazy dog.",
        "ru": "Быстрая коричневая лиса прыгает через ленивую собаку.",
        "fr": "Le renard brun rapide saute par-dessus le chien paresseux.",
        "de": "Der schnelle braune Fuchs springt ueber den faulen Hund.",
        "es": "El rapido zorro marron salta sobre el perro perezoso.",
        "ar": "الثعلب البني السريع يقفز فوق الكلب الكسول.",
        "hi": "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।",
        "tr": "Hizli kahverengi tilki tembel kopegin uzerinden atlar.",
        "pt": "A rapida raposa marrom pula sobre o cachorro preguicoso.",
        "uk": "Швидка коричнева лисиця стрибає через лінивого пса.",
        "zh": "快速的棕色狐狸跳过了懒惰的狗。",
        "ja": "素早い茶色のキツネが怠惰な犬を飛び越える。",
        "ko": "빠른 갈색 여우가 게으른 개를 뛰어넘습니다.",
    }

    print(f"\n--- Tokenizer Verification ---")
    print(f"Vocab size: {sp.get_piece_size():,}\n")

    for lang_code in languages:
        text = test_strings.get(lang_code, f"Test text for {lang_code}")
        tokens = sp.encode(text)
        pieces = sp.encode(text, out_type=str)
        ratio = len(tokens) / len(text.split())

        print(f"  [{lang_code}] {len(tokens):3d} tokens | {ratio:.1f} tok/word | {text[:50]}...")
        if len(pieces) <= 20:
            print(f"        {pieces}")

    # Compare English efficiency: 32K vs new vocab
    en_text = "The model architecture uses grouped query attention with rotary position embeddings."
    en_tokens = sp.encode(en_text)
    print(f"\n  English efficiency: {len(en_tokens)} tokens for {len(en_text.split())} words ({len(en_tokens)/len(en_text.split()):.1f} tok/word)")
    print(f"  (Optimal: ~1.3 tok/word for English BPE)")


def main():
    parser = argparse.ArgumentParser(description="Train multilingual tokenizer for nanollama")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3],
                        help="Language tier (1=Latin+Cyrillic, 2=+Arabic/Devanagari, 3=+CJK)")
    parser.add_argument("--languages", type=str, default=None,
                        help="Comma-separated language codes (overrides --tier). Example: en,ru,fr")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="Vocabulary size (default: 48K/64K/96K per tier)")
    parser.add_argument("--samples-per-lang", type=int, default=100000,
                        help="Text samples per language from CulturaX (default: 100000)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ~/.cache/nanollama/tokenizer)")
    parser.add_argument("--fineweb-english", action="store_true", default=True,
                        help="Use FineWeb-Edu for English instead of CulturaX (default: true)")
    parser.add_argument("--no-fineweb-english", action="store_false", dest="fineweb_english",
                        help="Use CulturaX for English too")
    parser.add_argument("--verify-only", type=str, default=None,
                        help="Only verify an existing tokenizer.model file")
    parser.add_argument("--keep-text", action="store_true",
                        help="Keep the downloaded text file after training")

    args = parser.parse_args()

    # Determine languages
    if args.verify_only:
        tier = args.tier or 1
        languages = TIERS[tier]
        if args.languages:
            languages = {code.strip(): code.strip() for code in args.languages.split(",")}
        verify_tokenizer(args.verify_only, languages)
        return

    if args.languages:
        lang_codes = [c.strip() for c in args.languages.split(",")]
        all_langs = {**TIER_3}  # Use tier 3 as name lookup
        languages = {}
        for code in lang_codes:
            languages[code] = all_langs.get(code, code)
        tier = None
    elif args.tier:
        tier = args.tier
        languages = TIERS[tier]
    else:
        print("ERROR: Specify --tier or --languages")
        parser.print_help()
        sys.exit(1)

    # Determine vocab size
    if args.vocab_size:
        vocab_size = args.vocab_size
    elif tier:
        vocab_size = DEFAULT_VOCAB[tier]
    else:
        vocab_size = 48000  # Default for custom language lists

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        base_dir = os.environ.get("NANOLLAMA_DIR", os.path.expanduser("~/.cache/nanollama"))
        tier_suffix = f"_tier{tier}" if tier else f"_{len(languages)}lang"
        output_dir = os.path.join(base_dir, f"tokenizer{tier_suffix}")

    print("=" * 60)
    print("  nanollama Multilingual Tokenizer Training")
    print("=" * 60)
    print(f"\n  Tier: {tier or 'custom'}")
    print(f"  Languages: {', '.join(f'{code} ({name})' for code, name in languages.items())}")
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  Samples per language: {args.samples_per_lang:,}")
    print(f"  Output: {output_dir}")
    print()

    # Step 1: Download text data
    text_file = os.path.join(tempfile.gettempdir(), f"nanollama_tokenizer_train.txt")

    # Download non-English from CulturaX
    non_en_langs = {k: v for k, v in languages.items() if k != "en"}
    if non_en_langs:
        download_culturax(non_en_langs, args.samples_per_lang, text_file)
    else:
        # Create empty file
        open(text_file, "w").close()

    # Download English (FineWeb-Edu or CulturaX)
    if "en" in languages:
        if args.fineweb_english:
            download_fineweb_english(args.samples_per_lang, text_file)
        else:
            # Append English from CulturaX
            download_culturax({"en": "English"}, args.samples_per_lang, text_file + ".en")
            with open(text_file, "a") as out, open(text_file + ".en") as inp:
                out.write(inp.read())
            os.unlink(text_file + ".en")

    # Step 2: Train SentencePiece
    sp = train_sentencepiece(
        input_file=text_file,
        output_dir=output_dir,
        vocab_size=vocab_size,
        num_languages=len(languages),
    )

    # Step 3: Verify
    model_path = os.path.join(output_dir, "tokenizer.model")
    verify_tokenizer(model_path, languages)

    # Cleanup
    if not args.keep_text:
        os.unlink(text_file)
        print(f"\nCleaned up temp file.")

    total_vocab = sp.get_piece_size() + len(SPECIAL_TOKENS)
    print(f"\n{'=' * 60}")
    print(f"  Tokenizer saved to: {output_dir}")
    print(f"  Total vocab: {total_vocab:,} ({sp.get_piece_size():,} BPE + {len(SPECIAL_TOKENS)} special)")
    print(f"  Files: tokenizer.model, tokenizer.vocab, special_tokens.txt")
    print(f"\n  Use in training:")
    print(f"    python -m scripts.base_train --depth 24 --tokenizer {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
