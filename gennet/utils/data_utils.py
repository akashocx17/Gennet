"""Data utility functions."""

import torch
from typing import Dict, Optional


def create_dummy_batch(
    batch_size: int = 4,
    seq_length: int = 128,
    image_size: int = 224,
    num_classes: Optional[int] = 10,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Create a dummy batch for testing.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length for text
        image_size: Image size
        num_classes: Number of classes (None for no labels)
        device: Device to create tensors on
        
    Returns:
        Dictionary with dummy batch data
    """
    batch = {
        'text_input_ids': torch.randint(0, 30000, (batch_size, seq_length), device=device),
        'text_attention_mask': torch.ones(batch_size, seq_length, device=device),
        'pixel_values': torch.randn(batch_size, 3, image_size, image_size, device=device),
    }
    
    if num_classes is not None:
        batch['labels'] = torch.randint(0, num_classes, (batch_size,), device=device)
        batch['rewards'] = torch.randn(batch_size, 1, device=device)
    
    return batch
