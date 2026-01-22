# Gennet - Multi-Modal Neural Network with Reinforcement Learning

A PyTorch-based framework for training multi-modal neural networks that combine text and vision encoders with reinforcement learning capabilities for reasoning.

## Features

- **Multi-Lingual Text Processing**: ModernBERT encoder with:
  - Domain-Adaptive Pre-Training (DAP) via MLM
  - CLS token fine-tuning support
  - Multi-lingual capabilities

- **Vision Processing**: Siglip2 vision encoder
  - Vision-only encoder mode
  - Pre-trained on large-scale image-text pairs

- **Multi-Modal Fusion**: MLP-based cross-modal fusion
  - Cross-attention mechanisms
  - Residual connections
  - Layer normalization

- **Reinforcement Learning**: Actor-Critic architecture
  - Policy gradient methods
  - Value network for baseline
  - Action reasoning embeddings

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Multi-Modal Model                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  ModernBERT      │         │  Siglip2 Vision  │          │
│  │  Text Encoder    │         │  Encoder         │          │
│  │  + DAP (MLM)     │         │  (Vision Only)   │          │
│  │  + CLS Token     │         │                  │          │
│  └────────┬─────────┘         └────────┬─────────┘          │
│           │                            │                     │
│           └────────────┬───────────────┘                     │
│                        │                                     │
│                ┌───────▼────────┐                            │
│                │  Cross-Modal   │                            │
│                │  Fusion (MLP)  │                            │
│                │  + Attention   │                            │
│                └───────┬────────┘                            │
│                        │                                     │
│                ┌───────▼────────┐                            │
│                │  RL Layer      │                            │
│                │  Actor-Critic  │                            │
│                │  + Reasoning   │                            │
│                └───────┬────────┘                            │
│                        │                                     │
│                  ┌─────▼──────┐                              │
│                  │  Output    │                              │
│                  └────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/akashocx17/Gennet.git
cd Gennet

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
import torch
from gennet import MultiModalModel, MultiModalTrainer
from gennet.configs.model_config import ModelConfig

# Create model configuration
config = ModelConfig(num_classes=10)

# Initialize model
model = MultiModalModel(config)

# Create trainer
trainer = MultiModalTrainer(model, config)
trainer.configure_optimizers(learning_rate=1e-4)

# Prepare your data (text_input_ids, text_attention_mask, pixel_values, labels)
# Then train
trainer.train_epoch(train_dataloader)

# Optional: Domain-Adaptive Pre-Training (MLM)
from torch.utils.data import DataLoader
domain_texts = [
  "Clinical findings indicate elevated markers <DOMAIN_A>.",
  "Financial report shows increased revenue <DOMAIN_B>."
]
text_loader = DataLoader(domain_texts, batch_size=2, shuffle=True)
trainer.domain_adaptive_pretrain_mlm(
  text_dataloader=text_loader,
  epochs=1,
  special_tokens=["<DOMAIN_A>", "<DOMAIN_B>"]
)
```

## Usage Example

See `examples/train_example.py` for a complete training example:

```bash
cd examples
python train_example.py
```

## Configuration

The model can be configured through `ModelConfig`:

```python
from gennet.configs.model_config import (
    ModelConfig,
    ModernBERTConfig,
    Siglip2Config,
    FusionConfig,
    RLConfig
)

config = ModelConfig(
    text_config=ModernBERTConfig(
        model_name="answerdotai/ModernBERT-base",
        use_cls_token=True,
        finetune=True,
        use_domain_adaptive_pretraining=True,
        dap_special_tokens=["<DOMAIN_A>", "<DOMAIN_B>"],
        dap_mlm_epochs=1,
        dap_learning_rate=5e-5
    ),
    vision_config=Siglip2Config(
        model_name="google/siglip-base-patch16-224",
        use_vision_only=True
    ),
    fusion_config=FusionConfig(
        fusion_hidden_size=1024,
        num_fusion_layers=2
    ),
    rl_config=RLConfig(
        num_actions=10,
        use_policy_gradient=True,
        use_value_network=True
    ),
    num_classes=10
)
```

## Components

### 1. Text Encoder (ModernBERT)
- Multilingual support
- Domain-Adaptive Pre-Training (MLM) support
- CLS token extraction
- Fine-tuning support

### 2. Vision Encoder (Siglip2)
- Vision-only mode
- Pre-trained features
- Pooled and sequence outputs

### 3. Fusion Layer
- Cross-modal attention
- MLP-based fusion
- Residual connections

### 4. RL Layer
- Actor-Critic architecture
- Policy gradient methods
- Value estimation
- Action reasoning

## Training

The trainer supports:
- Supervised classification
- Reinforcement learning with policy gradients
- Mixed training strategies
- Checkpoint saving/loading
- Evaluation metrics

```python
# Training with RL
trainer.train_epoch(train_loader, use_rl_loss=True)

# Evaluation
metrics = trainer.evaluate(val_loader)

# Save checkpoint
trainer.save_checkpoint("checkpoint.pt", epoch=1)
```

## Model Components

### ModernBERTEncoder
Handles text encoding with standard pooling and pre-training utilities:
- Standard CLS token extraction
- Mean pooling
- Special token utilities for DAP (MLM)
- Configurable fine-tuning

### Siglip2VisionEncoder
Processes images using Siglip2's vision encoder:
- Vision-only mode for efficient processing
- Pre-trained on large-scale datasets
- Multiple output formats

### CrossModalFusion
Fuses text and vision features:
- Cross-attention between modalities
- MLP-based transformation
- Residual connections for better gradient flow

### ReinforcementLearningLayer
Enables reasoning capabilities:
- Policy network (Actor) for action selection
- Value network (Critic) for state evaluation
- Action embeddings for reasoning

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.35.0
- See `requirements.txt` for full list

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gennet2024,
  title={Gennet: Multi-Modal Neural Networks with Reinforcement Learning},
  author={Gennet Contributors},
  year={2024},
  url={https://github.com/akashocx17/Gennet}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- ModernBERT by Answer.AI
- Siglip2 by Google Research
- PyTorch and Hugging Face Transformers teams
