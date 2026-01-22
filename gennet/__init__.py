"""
Gennet - Multi-modal Neural Network with Reinforcement Learning

A framework for training multi-modal neural networks combining text (ModernBERT)
and vision (Siglip2) with reinforcement learning capabilities.
"""

__version__ = "0.1.0"

from gennet.models.multimodal_model import MultiModalModel
from gennet.training.trainer import MultiModalTrainer

__all__ = ["MultiModalModel", "MultiModalTrainer"]
