"""
ARC (AI2 Reasoning Challenge) task for nanollama.
Multiple choice science questions.
"""

import os
import json
from typing import Dict, Iterator

import torch
import torch.nn.functional as F

from tasks.common import Task, TaskSample
from nanollama.common import download_file_with_lock, print0


class ARCTask(Task):
    """ARC Challenge evaluation task."""
    
    name = "arc"
    
    def __init__(self, split: str = "test", num_samples: int = None):
        super().__init__(num_samples)
        self.split = split
        self._load_data()
    
    def _load_data(self):
        """Load ARC dataset."""
        # Placeholder - actual implementation would download from HuggingFace
        # e.g.: datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split=self.split)
        self.samples = []
        # Note: samples will be empty until dataset loading is implemented
    
    def get_samples(self) -> Iterator[TaskSample]:
        """Yield ARC samples."""
        count = 0
        for sample in self.samples:
            if self.num_samples and count >= self.num_samples:
                break
            
            question = sample.get("question", "")
            choices = sample.get("choices", {}).get("text", [])
            answer_key = sample.get("answerKey", "A")
            
            # Convert answer key to index
            answer_idx = ord(answer_key) - ord('A')
            
            yield TaskSample(
                prompt=question,
                choices=choices,
                answer=choices[answer_idx] if answer_idx < len(choices) else "",
                answer_idx=answer_idx,
            )
            count += 1
    
    def evaluate(self, model, tokenizer, device) -> Dict[str, float]:
        """Evaluate model on ARC."""
        model.eval()
        correct = 0
        total = 0
        
        for sample in self.get_samples():
            if not sample.choices:
                continue
            
            # Compute log probability for each choice
            choice_scores = []
            
            for choice in sample.choices:
                prompt = f"Question: {sample.prompt}\nAnswer: {choice}"
                tokens = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
                
                with torch.no_grad():
                    input_ids = torch.tensor([tokens[:-1]], device=device)
                    target_ids = torch.tensor([tokens[1:]], device=device)
                    logits = model(input_ids)
                    
                    # Compute log probability
                    log_probs = F.log_softmax(logits, dim=-1)
                    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
                    score = token_log_probs.sum().item()
                
                choice_scores.append(score)
            
            # Check if best choice matches answer
            predicted_idx = max(range(len(choice_scores)), key=lambda i: choice_scores[i])
            if predicted_idx == sample.answer_idx:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
