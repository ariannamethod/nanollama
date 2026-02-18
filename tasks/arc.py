"""
The ARC dataset from Allen AI.
https://huggingface.co/datasets/allenai/ai2_arc

Adapted from nanochat for nanollama.
"""

from datasets import load_dataset
from tasks.common import Task, render_mc


class ARC(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], "ARC subset must be ARC-Easy or ARC-Challenge"
        assert split in ["train", "validation", "test"], "ARC split must be train|validation|test"
        self.ds = load_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        choices = row["choices"]["text"]
        answer_string = row["answerKey"]
        letters = row["choices"]["label"]
        assert answer_string in letters
        user_message = render_mc(question, letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string}
        ]
        conversation = {
            "messages": messages,
            "letters": letters,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in conversation['letters']
        assistant_message = conversation['messages'][-1]['content']
        return assistant_response == assistant_message
