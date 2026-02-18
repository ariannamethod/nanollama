"""
CustomJSON task for loading conversations from JSONL files.
Each line in the JSONL file should be a JSON array of messages.

Adapted from nanochat for nanollama.
"""

import os
import json
from tasks.common import Task


class CustomJSON(Task):
    """
    Load conversations from a JSONL file.
    Each line should be a JSON array of message objects with 'role' and 'content' fields.
    Example line: [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]
    """

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} does not exist")
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    messages = json.loads(line)
                    assert isinstance(messages, list)
                    assert len(messages) >= 2
                    for i, message in enumerate(messages):
                        assert "role" in message
                        assert "content" in message
                        expected_role = "user" if i % 2 == 0 else "assistant"
                        assert message["role"] == expected_role
                        assert isinstance(message["content"], str)
                    self.conversations.append(messages)

        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        conversation = {"messages": messages}
        return conversation
