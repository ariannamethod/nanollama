"""
Common task utilities for nanollama evaluation.
"""

from typing import List, Dict, Any, Iterator
from dataclasses import dataclass


@dataclass
class TaskSample:
    """A single evaluation sample."""
    prompt: str
    choices: List[str] = None  # For multiple choice
    answer: str = None  # Correct answer
    answer_idx: int = None  # Index for multiple choice
    metadata: Dict[str, Any] = None


class Task:
    """Base class for evaluation tasks."""
    
    name: str = "base"
    
    def __init__(self, num_samples: int = None):
        self.num_samples = num_samples
    
    def get_samples(self) -> Iterator[TaskSample]:
        """Yield task samples."""
        raise NotImplementedError
    
    def evaluate(self, model, tokenizer, device) -> Dict[str, float]:
        """Evaluate model on this task."""
        raise NotImplementedError


class TaskMixture:
    """Mixture of multiple tasks."""
    
    def __init__(self, tasks: List[Task], weights: List[float] = None):
        self.tasks = tasks
        self.weights = weights or [1.0] * len(tasks)
    
    def evaluate(self, model, tokenizer, device) -> Dict[str, float]:
        """Evaluate model on all tasks."""
        results = {}
        for task in self.tasks:
            task_results = task.evaluate(model, tokenizer, device)
            for key, value in task_results.items():
                results[f"{task.name}/{key}"] = value
        return results


class TaskSequence:
    """Sequential evaluation of tasks."""
    
    def __init__(self, tasks: List[Task]):
        self.tasks = tasks
    
    def evaluate(self, model, tokenizer, device) -> Dict[str, float]:
        """Evaluate model on tasks in sequence."""
        results = {}
        for task in self.tasks:
            task_results = task.evaluate(model, tokenizer, device)
            results.update(task_results)
        return results
