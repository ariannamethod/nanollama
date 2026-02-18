"""
Evaluate the Chat model on HumanEval dataset.
This is a coding benchmark (has nothing to do with humans).

Adapted from nanochat for nanollama.
"""

import re
from datasets import load_dataset
from tasks.common import Task


def extract_imports(prompt):
    """Extract import statements from the beginning of a code block."""
    imports = []
    for line in prompt.split('\n'):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            imports.append(stripped)
        elif stripped and not stripped.startswith('#'):
            break
    return '\n'.join(imports)


def extract_program(completion):
    """
    Extract Python code from LLM completion.
    Handles code wrapped in ```python ... ``` blocks or plain code.
    """
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, completion, re.DOTALL)
    if matches:
        return matches[0].strip()
    return completion.strip()


def execute_code_safe(program: str, timeout: int = 5) -> bool:
    """
    Execute code safely with timeout. Returns True if code runs without error.
    """
    import subprocess
    import tempfile
    import os
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(program)
            f.flush()
            temp_path = f.name
        
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            timeout=timeout,
            text=True
        )
        os.unlink(temp_path)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        try:
            os.unlink(temp_path)
        except:
            pass
        return False


class HumanEval(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        prompt = row['prompt']
        solution = row['canonical_solution']
        entry_point = row['entry_point']
        test = row['test']
        complete_solution = f"{prompt}\n{solution}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": complete_solution},
        ]
        conversation = {
            "messages": messages,
            "entry_point": entry_point,
            "test": test,
        }
        return conversation

    def evaluate(self, conversation, completion):
        """Given (conversation, completion), return boolean success."""
        imports = extract_imports(conversation['messages'][0]['content'])
        completion_code = extract_program(completion)
        program = (
            imports
            + "\n\n"
            + completion_code
            + "\n\n"
            + conversation['test']
            + "\n"
            + f"check({conversation['entry_point']})"
        )
        success = execute_code_safe(program)
        return success
