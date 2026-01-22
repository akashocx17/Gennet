"""Model configuration for multi-modal neural network."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModernBERTConfig:
    """Configuration for ModernBERT text encoder."""
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    hidden_size: int = 768
    use_cls_token: bool = True
    use_dap: bool = True  # Discriminative Adapter Pooling
    finetune: bool = True


@dataclass
class Siglip2Config:
    """Configuration for Siglip2 vision encoder."""
    model_name: str = "google/siglip-base-patch16-224"
    image_size: int = 224
    hidden_size: int = 768
    use_vision_only: bool = True


@dataclass
class FusionConfig:
    """Configuration for multi-modal fusion layer."""
    fusion_hidden_size: int = 1024
    num_fusion_layers: int = 2
    dropout: float = 0.1
    activation: str = "gelu"


@dataclass
class RLConfig:
    """Configuration for Reinforcement Learning layer."""
    num_actions: int = 10
    gamma: float = 0.99  # Discount factor
    learning_rate: float = 1e-4
    use_policy_gradient: bool = True
    use_value_network: bool = True


@dataclass
class ModelConfig:
    """Main configuration for the multi-modal model."""
    text_config: ModernBERTConfig = None
    vision_config: Siglip2Config = None
    fusion_config: FusionConfig = None
    rl_config: RLConfig = None
    num_classes: Optional[int] = None
    
    def __post_init__(self):
        if self.text_config is None:
            self.text_config = ModernBERTConfig()
        if self.vision_config is None:
            self.vision_config = Siglip2Config()
        if self.fusion_config is None:
            self.fusion_config = FusionConfig()
        if self.rl_config is None:
            self.rl_config = RLConfig()
