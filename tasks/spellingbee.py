"""
Spelling tasks for nanollama.

Two tasks:
1. SpellingBee: Counting the number of occurrences of a letter in a word
2. SimpleSpelling: Simply spelling words

Adapted from nanochat for nanollama.
"""

import re
import random
from tasks.common import Task
from nanollama.common import download_file_with_lock

LETTERS = "abcdefghijklmnopqrstuvwxyz"
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
TEST_RANDOM_SEED_OFFSET = 10_000_000

ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(completion):
    """Extract the numerical answer after #### marker."""
    match = ANSWER_RE.search(completion)
    if match:
        match_str = match.group(1).strip().replace(",", "")
        return match_str
    return None


USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
]


class SpellingBee(Task):

    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee split must be train|test"
        self.size = size
        self.split = split
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        self.words = words

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == 'train' else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)

        word = rng.choice(self.words)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)
        count = word.count(letter)

        template = rng.choice(USER_MSG_TEMPLATES)
        if rng.random() < 0.3:
            template = template.lower()
        quote_options = ['', "'", '"']
        letter_quote = rng.choice(quote_options)
        word_quote = rng.choice(quote_options)
        letter_wrapped = f"{letter_quote}{letter}{letter_quote}"
        word_wrapped = f"{word_quote}{word}{word_quote}"
        user_msg = template.format(letter=letter_wrapped, word=word_wrapped)
        if rng.random() < 0.5:
            user_msg += "?"

        # Build assistant response with manual counting + Python verification
        assistant_parts = []
        word_letters = ",".join(list(word))
        manual_text = f"""We are asked to find the number '{letter}' in the word '{word}'. Let me try a manual approach first.

First spell the word out:
{word}:{word_letters}

Then count the occurrences of '{letter}':
"""
        running_count = 0
        for i, char in enumerate(word, 1):
            if char == letter:
                running_count += 1
                manual_text += f"{i}:{char} hit! count={running_count}\n"
            else:
                manual_text += f"{i}:{char}\n"

        manual_text += f"\nThis gives us {running_count}."
        assistant_parts.append({"type": "text", "text": manual_text})
        assistant_parts.append({"type": "text", "text": "\n\nLet me double check this using Python:\n\n"})
        python_expr = f"'{word}'.count('{letter}')"
        assistant_parts.append({"type": "python", "text": python_expr})
        assistant_parts.append({"type": "python_output", "text": str(count)})
        assistant_parts.append({"type": "text", "text": f"\n\nPython gives us {count}.\n\nMy final answer is:\n\n#### {count}"})

        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_parts}
        ]
        conversation = {"messages": messages}
        return conversation

    def evaluate(self, conversation, assistant_response):
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant"
        assert isinstance(assistant_message['content'], list)
        last_text_part = assistant_message['content'][-1]['text']
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        is_correct = self.evaluate(conversation, assistant_response)
        return float(is_correct)


class SimpleSpelling(Task):
    """Simple task to practice spelling words."""

    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"]
        self.size = size
        self.split = split
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        rng = random.Random(42)
        rng.shuffle(words)
        self.words = words

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == 'train' else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)
        word = rng.choice(self.words)
        word_letters = ",".join(list(word))
        messages = [
            {"role": "user", "content": f"Spell the word: {word}"},
            {"role": "assistant", "content": f"{word}:{word_letters}"}
        ]
        conversation = {"messages": messages}
        return conversation
