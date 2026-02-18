"""
SentencePiece-based tokenizer for nanollama.
Supports Llama 3 chat format and tool use.

Unlike nanochat's GPT-4 style BPE, nanollama uses SentencePiece BPE
which is the standard for Llama models. Users can also plug in
Llama 3's original tokenizer.
"""

import os
import copy
from functools import lru_cache
from typing import List, Optional, Union, Tuple

# Special tokens for nanollama
SPECIAL_TOKENS = [
    # Document delimiter
    "<|bos|>",
    # Chat tokens (Llama 3 style)
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    # Legacy nanochat-style tokens (for compatibility)
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    # Tool use
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

# Default split pattern for BPE
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class SentencePieceTokenizer:
    """
    SentencePiece-based tokenizer for nanollama.
    
    This tokenizer uses SentencePiece BPE which is standard for Llama models.
    It supports both training from scratch and loading from a pretrained model.
    """
    
    def __init__(self, sp_model, special_tokens: dict):
        """
        Initialize tokenizer with a SentencePiece model.
        
        Args:
            sp_model: Loaded SentencePiece processor (sentencepiece.SentencePieceProcessor)
            special_tokens: Dict mapping special token strings to their IDs
        """
        self.sp_model = sp_model
        self.special_tokens = special_tokens
        self._special_tokens_set = set(special_tokens.keys())
        self.bos_token_id = special_tokens.get("<|bos|>", special_tokens.get("<|begin_of_text|>", 1))
    
    @classmethod
    def from_directory(cls, tokenizer_dir: str):
        """Load tokenizer from a directory."""
        import sentencepiece as spm
        
        model_path = os.path.join(tokenizer_dir, "tokenizer.model")
        sp_model = spm.SentencePieceProcessor(model_file=model_path)
        
        # Load special tokens mapping
        special_tokens_path = os.path.join(tokenizer_dir, "special_tokens.txt")
        special_tokens = {}
        if os.path.exists(special_tokens_path):
            with open(special_tokens_path, 'r') as f:
                for line in f:
                    token, idx = line.strip().split('\t')
                    special_tokens[token] = int(idx)
        else:
            # Default special tokens mapping
            vocab_size = sp_model.get_piece_size()
            for i, token in enumerate(SPECIAL_TOKENS):
                special_tokens[token] = vocab_size + i
        
        return cls(sp_model, special_tokens)
    
    @classmethod
    def from_pretrained(cls, model_name: str):
        """Load a pretrained Llama tokenizer."""
        # Support loading Llama 3 tokenizer from HuggingFace
        try:
            from transformers import AutoTokenizer
            hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Create wrapper that mimics our interface
            return HuggingFaceTokenizerWrapper(hf_tokenizer)
        except ImportError:
            raise ImportError("transformers library required for loading pretrained tokenizers")
    
    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size: int = 32000, output_dir: str = None):
        """
        Train a new SentencePiece tokenizer from an iterator of texts.
        
        Args:
            text_iterator: Iterator yielding text strings
            vocab_size: Target vocabulary size (default: 32000)
            output_dir: Directory to save the trained model
        """
        import sentencepiece as spm
        import tempfile
        
        # Create temporary file for training data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for text in text_iterator:
                f.write(text + '\n')
            train_file = f.name
        
        # Train SentencePiece model
        model_prefix = os.path.join(output_dir or tempfile.gettempdir(), "tokenizer")
        
        spm.SentencePieceTrainer.train(
            input=train_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size - len(SPECIAL_TOKENS),  # Reserve space for special tokens
            model_type='bpe',
            character_coverage=0.9995,
            num_threads=os.cpu_count(),
            split_digits=True,
            byte_fallback=True,
        )
        
        # Load trained model
        sp_model = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
        
        # Add special tokens
        base_vocab_size = sp_model.get_piece_size()
        special_tokens = {}
        for i, token in enumerate(SPECIAL_TOKENS):
            special_tokens[token] = base_vocab_size + i
        
        # Save special tokens mapping
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "special_tokens.txt"), 'w') as f:
                for token, idx in special_tokens.items():
                    f.write(f"{token}\t{idx}\n")
        
        # Clean up temp file
        os.unlink(train_file)
        
        return cls(sp_model, special_tokens)
    
    def get_vocab_size(self) -> int:
        """Return total vocabulary size including special tokens."""
        return self.sp_model.get_piece_size() + len(self.special_tokens)
    
    def get_special_tokens(self) -> set:
        """Return set of special token strings."""
        return self._special_tokens_set
    
    @lru_cache(maxsize=32)
    def encode_special(self, text: str) -> Optional[int]:
        """Encode a single special token."""
        return self.special_tokens.get(text)
    
    def get_bos_token_id(self) -> int:
        """Return the BOS token ID."""
        return self.bos_token_id
    
    def encode(
        self,
        text: Union[str, List[str]],
        prepend: Optional[Union[str, int]] = None,
        append: Optional[Union[str, int]] = None,
        num_threads: int = 8,
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode text to token IDs.
        
        Args:
            text: String or list of strings to encode
            prepend: Token to prepend (string or ID)
            append: Token to append (string or ID)
            num_threads: Number of threads for batch encoding
        
        Returns:
            List of token IDs or list of lists for batch input
        """
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        else:
            prepend_id = None
            
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
        else:
            append_id = None
        
        if isinstance(text, str):
            ids = self.sp_model.encode(text)
            if prepend_id is not None:
                ids.insert(0, prepend_id)
            if append_id is not None:
                ids.append(append_id)
            return ids
        else:
            # Batch encoding
            all_ids = self.sp_model.encode(text)
            if prepend_id is not None or append_id is not None:
                for ids in all_ids:
                    if prepend_id is not None:
                        ids.insert(0, prepend_id)
                    if append_id is not None:
                        ids.append(append_id)
            return all_ids
    
    def __call__(self, *args, **kwargs):
        """Alias for encode."""
        return self.encode(*args, **kwargs)
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        # Filter out special tokens and decode
        regular_ids = [i for i in ids if i < self.sp_model.get_piece_size()]
        text = self.sp_model.decode(regular_ids)
        
        # Insert special token strings where they appear
        result = []
        last_pos = 0
        for i, token_id in enumerate(ids):
            if token_id >= self.sp_model.get_piece_size():
                # Find the special token string
                for token_str, tid in self.special_tokens.items():
                    if tid == token_id:
                        result.append(token_str)
                        break
        
        return text
    
    def save(self, tokenizer_dir: str):
        """Save tokenizer to directory."""
        os.makedirs(tokenizer_dir, exist_ok=True)
        
        # Save SentencePiece model (need to copy from temp location)
        # This assumes the model file exists
        import shutil
        model_path = os.path.join(tokenizer_dir, "tokenizer.model")
        if hasattr(self.sp_model, 'model_file'):
            shutil.copy(self.sp_model.model_file, model_path)
        
        # Save special tokens
        with open(os.path.join(tokenizer_dir, "special_tokens.txt"), 'w') as f:
            for token, idx in self.special_tokens.items():
                f.write(f"{token}\t{idx}\n")
        
        print(f"Saved tokenizer to {tokenizer_dir}")
    
    def render_conversation_llama3(self, conversation: dict, max_tokens: int = 2048) -> Tuple[List[int], List[int]]:
        """
        Render a conversation in Llama 3 chat format.
        
        Llama 3 format:
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        {message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        {response}<|eot_id|>
        
        Returns:
            ids: Token IDs
            mask: Training mask (1 for tokens to train on, 0 otherwise)
        """
        ids, mask = [], []
        
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))
        
        # Get special tokens
        begin_of_text = self.encode_special("<|begin_of_text|>")
        start_header = self.encode_special("<|start_header_id|>")
        end_header = self.encode_special("<|end_header_id|>")
        eot = self.encode_special("<|eot_id|>")
        
        messages = conversation.get("messages", [])
        
        # Handle system message
        if messages and messages[0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            if len(messages) > 1:
                messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        
        # Begin text
        add_tokens(begin_of_text, 0)
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            # Header
            add_tokens(start_header, 0)
            add_tokens(self.encode(role), 0)
            add_tokens(end_header, 0)
            add_tokens(self.encode("\n"), 0)
            
            # Content
            content_ids = self.encode(content) if isinstance(content, str) else []
            mask_val = 1 if role == "assistant" else 0
            add_tokens(content_ids, mask_val)
            
            # End of turn
            add_tokens(eot, mask_val if role == "assistant" else 0)
        
        # Truncate
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        
        return ids, mask
    
    def render_conversation(self, conversation: dict, max_tokens: int = 2048) -> Tuple[List[int], List[int]]:
        """
        Render conversation in nanochat-compatible format.
        
        Returns:
            ids: Token IDs
            mask: Training mask
        """
        ids, mask = [], []
        
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))
        
        messages = conversation.get("messages", [])
        
        # Handle system message
        if messages and messages[0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            if len(messages) > 1:
                messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        
        # Special tokens
        bos = self.get_bos_token_id()
        user_start = self.encode_special("<|user_start|>")
        user_end = self.encode_special("<|user_end|>")
        assistant_start = self.encode_special("<|assistant_start|>")
        assistant_end = self.encode_special("<|assistant_end|>")
        
        add_tokens(bos, 0)
        
        for i, message in enumerate(messages):
            content = message["content"]
            
            if message["role"] == "user":
                add_tokens(user_start, 0)
                add_tokens(self.encode(content), 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                add_tokens(self.encode(content), 1)
                add_tokens(assistant_end, 1)
        
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        
        return ids, mask


class HuggingFaceTokenizerWrapper:
    """Wrapper around HuggingFace tokenizer to provide consistent interface."""
    
    def __init__(self, hf_tokenizer):
        self.hf_tokenizer = hf_tokenizer
        self.bos_token_id = hf_tokenizer.bos_token_id or 1
    
    def get_vocab_size(self) -> int:
        return len(self.hf_tokenizer)
    
    def get_bos_token_id(self) -> int:
        return self.bos_token_id
    
    def encode(self, text, prepend=None, append=None, **kwargs):
        ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.hf_tokenizer.convert_tokens_to_ids(prepend)
            ids.insert(0, prepend_id)
        if append is not None:
            append_id = append if isinstance(append, int) else self.hf_tokenizer.convert_tokens_to_ids(append)
            ids.append(append_id)
        return ids
    
    def decode(self, ids):
        return self.hf_tokenizer.decode(ids)
    
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


# Convenience functions

def get_tokenizer():
    """Get the default tokenizer from cache directory."""
    from nanollama.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return SentencePieceTokenizer.from_directory(tokenizer_dir)
