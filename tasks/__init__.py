"""
Evaluation tasks for nanollama.
"""

from tasks.common import Task, TaskMixture, TaskSequence, render_mc
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.humaneval import HumanEval
from tasks.spellingbee import SpellingBee, SimpleSpelling
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON

__all__ = [
    'Task', 'TaskMixture', 'TaskSequence', 'render_mc',
    'ARC', 'GSM8K', 'MMLU', 'HumanEval', 
    'SpellingBee', 'SimpleSpelling', 'SmolTalk', 'CustomJSON',
]
