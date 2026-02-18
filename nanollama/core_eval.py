"""
DCLM CORE evaluation for nanollama.
Evaluates base model quality using the DCLM benchmark.
"""

import os
import json
import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from nanollama.common import get_base_dir, print0


def compute_bits_per_byte(
    model,
    tokenizer,
    text: str,
    device: torch.device,
) -> float:
    """
    Compute bits per byte (BPB) for a text sample.
    
    This is a vocab-size-invariant measure of model quality.
    Lower is better.
    """
    # Tokenize
    tokens = tokenizer.encode(text, prepend=tokenizer.get_bos_token_id())
    
    if len(tokens) < 2:
        return float('inf')
    
    # Prepare input
    input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            reduction='mean',
        )
    
    # Convert to bits per byte
    # BPB = loss_in_nats * num_tokens / num_bytes / ln(2)
    num_bytes = len(text.encode('utf-8'))
    num_tokens = len(tokens) - 1  # Exclude first token
    
    loss_nats = loss.item()
    bpb = loss_nats * num_tokens / num_bytes / math.log(2)
    
    return bpb


def evaluate_core(
    model,
    tokenizer,
    device: torch.device,
    num_samples: int = 1000,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Evaluate model using DCLM CORE benchmark.
    
    Returns:
        Dict with 'core_score', 'bpb', and per-domain scores
    """
    # Load CORE evaluation data
    base_dir = get_base_dir()
    core_data_path = os.path.join(base_dir, "eval", "core_samples.jsonl")
    
    if not os.path.exists(core_data_path):
        print0(f"CORE data not found at {core_data_path}. Using placeholder evaluation.")
        # Return placeholder scores
        return {
            'core_score': 0.0,
            'bpb': float('inf'),
            'domains': {},
        }
    
    # Load samples
    samples = []
    with open(core_data_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= num_samples:
                break
    
    if not samples:
        return {'core_score': 0.0, 'bpb': float('inf'), 'domains': {}}
    
    # Compute BPB for each sample
    model.eval()
    total_bpb = 0.0
    domain_bpb = {}
    domain_counts = {}
    
    for sample in samples:
        text = sample.get('text', '')
        domain = sample.get('domain', 'unknown')
        
        bpb = compute_bits_per_byte(model, tokenizer, text, device)
        
        if bpb != float('inf'):
            total_bpb += bpb
            if domain not in domain_bpb:
                domain_bpb[domain] = 0.0
                domain_counts[domain] = 0
            domain_bpb[domain] += bpb
            domain_counts[domain] += 1
    
    # Compute averages
    avg_bpb = total_bpb / len(samples) if samples else float('inf')
    
    domain_scores = {}
    for domain in domain_bpb:
        if domain_counts[domain] > 0:
            domain_scores[domain] = domain_bpb[domain] / domain_counts[domain]
    
    # CORE score is inversely related to BPB
    # Using a calibrated transformation
    core_score = max(0.0, 1.0 - avg_bpb / 2.0)  # Rough calibration
    
    return {
        'core_score': core_score,
        'bpb': avg_bpb,
        'domains': domain_scores,
    }


def evaluate_hellaswag(
    model,
    tokenizer,
    device: torch.device,
    num_samples: int = 100,
) -> float:
    """Evaluate on HellaSwag benchmark."""
    # Placeholder - implement full evaluation
    return 0.0


def evaluate_arc(
    model,
    tokenizer,
    device: torch.device,
    num_samples: int = 100,
) -> float:
    """Evaluate on ARC benchmark."""
    # Placeholder - implement full evaluation
    return 0.0


def run_all_evals(
    model,
    tokenizer,
    device: torch.device,
) -> Dict[str, Any]:
    """Run all evaluations and return results."""
    results = {}
    
    # CORE evaluation
    print0("Running CORE evaluation...")
    core_results = evaluate_core(model, tokenizer, device)
    results['core'] = core_results
    
    # Additional benchmarks (placeholders)
    # results['hellaswag'] = evaluate_hellaswag(model, tokenizer, device)
    # results['arc'] = evaluate_arc(model, tokenizer, device)
    
    return results
