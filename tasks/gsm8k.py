"""
GSM8K task for nanollama.
Grade school math word problems.
"""

import os
import json
import re
from typing import Dict, Iterator

import torch

from tasks.common import Task, TaskSample
from nanollama.common import print0


class GSM8KTask(Task):
    """GSM8K math evaluation task."""
    
    name = "gsm8k"
    
    def __init__(self, split: str = "test", num_samples: int = None):
        super().__init__(num_samples)
        self.split = split
        self._load_data()
    
    def _load_data(self):
        """Load GSM8K dataset."""
        self.samples = []
        print0(f"GSM8K task: {len(self.samples)} samples loaded")
    
    def get_samples(self) -> Iterator[TaskSample]:
        """Yield GSM8K samples."""
        count = 0
        for sample in self.samples:
            if self.num_samples and count >= self.num_samples:
                break
            
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            
            # Extract final numerical answer
            final_answer = self._extract_answer(answer)
            
            yield TaskSample(
                prompt=question,
                answer=final_answer,
                metadata={"full_answer": answer},
            )
            count += 1
    
    def _extract_answer(self, answer_text: str) -> str:
        """Extract final numerical answer from GSM8K answer format."""
        # GSM8K answers end with "#### <number>"
        match = re.search(r'####\s*([0-9,.\-]+)', answer_text)
        if match:
            return match.group(1).replace(',', '')
        return ""
    
    def _parse_generated_answer(self, text: str) -> str:
        """Parse numerical answer from generated text."""
        # Look for patterns like "the answer is X" or just the last number
        patterns = [
            r'[Tt]he answer is[:\s]*([0-9,.\-]+)',
            r'####\s*([0-9,.\-]+)',
            r'=\s*([0-9,.\-]+)\s*$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).replace(',', '')
        
        # Fall back to last number in text
        numbers = re.findall(r'[0-9,.\-]+', text)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return ""
    
    def evaluate(self, model, tokenizer, device, max_tokens: int = 256) -> Dict[str, float]:
        """Evaluate model on GSM8K."""
        from nanollama.engine import Engine
        
        model.eval()
        engine = Engine(model, tokenizer)
        
        correct = 0
        total = 0
        
        for sample in self.get_samples():
            prompt = f"Question: {sample.prompt}\nLet's solve this step by step.\n"
            tokens = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
            
            # Generate
            generated_tokens = []
            for token_column, _ in engine.generate(
                tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=0.0,
            ):
                generated_tokens.append(token_column[0])
            
            generated_text = tokenizer.decode(generated_tokens)
            predicted_answer = self._parse_generated_answer(generated_text)
            
            # Compare
            try:
                if float(predicted_answer) == float(sample.answer):
                    correct += 1
            except (ValueError, TypeError):
                pass
            
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
