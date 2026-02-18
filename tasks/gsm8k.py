"""
GSM8K task for nanollama.
Grade school math word problems.

Adapted from nanochat for nanollama.
"""

import re
from datasets import load_dataset
from tasks.common import Task

# Answer extraction regex
ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(completion):
    """Extract the numerical answer after #### marker."""
    match = ANSWER_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


class GSM8K(Task):
    """GSM8K math evaluation task."""

    def __init__(self, subset="main", split="train", **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], f"subset {subset} must be main|socratic"
        assert split in ["train", "test"], f"split {split} must be train|test"
        self.subset = subset
        self.split = split
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        answer = row["answer"]
        # Create conversation for training/evaluation
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        conversation = {"messages": messages}
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome.
        """
        assistant_message = conversation['messages'][-1]['content']
        ref_num = extract_answer(assistant_message)
        pred_num = extract_answer(assistant_response)
        is_correct = int(pred_num == ref_num) if ref_num and pred_num else 0
        return is_correct

    def reward(self, conversation, assistant_response):
        """Use simple 0-1 reward."""
        is_correct = self.evaluate(conversation, assistant_response)
        return float(is_correct)
