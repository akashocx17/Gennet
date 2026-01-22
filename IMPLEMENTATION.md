# Gennet Implementation Details

## Overview

Gennet is a multi-modal neural network framework that combines text and vision encoders with reinforcement learning capabilities. The implementation follows the requirements specified in the problem statement.

## Core Components

### 1. Text Encoder (ModernBERT)

**File:** `gennet/models/text_encoder.py`

**Features:**
- Uses multilingual ModernBERT from Answer.AI
- Supports Domain-Adaptive Pre-Training (DAP) via MLM (handled in trainer)
- Special token utilities for domain markers
- Supports CLS token extraction for fine-tuning
- Configurable fine-tuning mode

**Key Classes:**
- `ModernBERTEncoder`: Main encoder class that wraps ModernBERT with CLS support and DAP utilities (special tokens)

**Architecture:**
```
Input Text → Tokenizer → ModernBERT → Hidden States
                                           ↓
                        ┌──────────────────┴──────────────────┐
                        ↓                                      ↓
                   CLS Token                            Mean Pooling
                   (First Token)                         (Sequence Avg)
                        ↓                                      ↓
                Text Features                          Pooled Features
```

### 2. Vision Encoder (Siglip2)

**File:** `gennet/models/vision_encoder.py`

**Features:**
- Uses Siglip2 vision encoder from Google
- Vision-only mode (no text encoder)
- Pre-trained on large-scale image-text pairs
- Supports both raw PIL images and preprocessed pixel values

**Key Classes:**
- `Siglip2VisionEncoder`: Wraps Siglip2 vision model, extracts vision features

**Architecture:**
```
Input Images → Preprocessor → Siglip2 Vision Model → Vision Features
                                                            ↓
                                                    Pooled Output
```

### 3. Cross-Modal Fusion Layer

**File:** `gennet/models/fusion_layer.py`

**Features:**
- MLP-based fusion of text and vision features
- Cross-attention between modalities
- Residual connections
- Layer normalization

**Key Classes:**
- `CrossModalFusion`: Implements cross-modal attention and MLP fusion

**Architecture:**
```
Text Features ──────┐
                    ├──→ Project → Cross-Attention ──→ MLP Layers ──→ Fused Features
Vision Features ────┘                                       ↑
                                                            │
                                                    (Residual Connection)
```

**Process:**
1. Project text and vision features to same dimension
2. Apply cross-attention (text attends to vision, vision attends to text)
3. Concatenate attended features
4. Pass through MLP layers
5. Add residual connection and normalize

### 4. Reinforcement Learning Layer

**File:** `gennet/models/rl_layer.py`

**Features:**
- Actor-Critic architecture
- Policy gradient methods
- Value network for baseline
- Action reasoning embeddings

**Key Classes:**
- `PolicyNetwork`: Neural network that outputs action probabilities (Actor)
- `ValueNetwork`: Neural network that estimates state values (Critic)
- `ReinforcementLearningLayer`: Main RL component combining policy and value networks

**Architecture:**
```
                        Fused Features
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
              Policy Network      Value Network
                (Actor)             (Critic)
                    ↓                   ↓
           Action Probabilities    State Value
                    ↓
            Sample Action
                    ↓
          Action Embedding
                    ↓
      State + Action = Reasoned State
```

**RL Methods:**
- `forward()`: Performs forward pass, samples actions
- `compute_returns()`: Computes discounted returns from rewards
- `compute_policy_loss()`: Computes policy gradient loss with advantage
- `compute_value_loss()`: Computes MSE loss for value network

### 5. Multi-Modal Model

**File:** `gennet/models/multimodal_model.py`

**Features:**
- Integrates all components (text, vision, fusion, RL)
- End-to-end training
- Optional classification head
- Flexible configuration

**Key Classes:**
- `MultiModalModel`: Main model class that orchestrates all components

**Full Architecture:**
```
Text Input ──→ ModernBERT ──┐
                            ├──→ Fusion Layer ──→ RL Layer ──→ Output
Image Input ──→ Siglip2 ────┘                          ↓
                                              Action Reasoning
                                                       ↓
                                            Classification (optional)
```

## Training System

### Trainer

**File:** `gennet/training/trainer.py`

**Features:**
- Supervised classification training
- Reinforcement learning with policy gradients
- Mixed training strategies
- Checkpoint management
- Evaluation metrics

**Key Classes:**
- `MultiModalTrainer`: Main training class

**Training Modes:**
1. **Supervised Only**: Uses classification loss
2. **RL Only**: Uses policy gradient and value losses
3. **Combined**: Uses both supervised and RL losses

**Loss Functions:**
- Classification Loss: Cross-entropy
- RL Policy Loss: Policy gradient with advantage
- RL Value Loss: MSE between predicted and actual returns

## Configuration System

**File:** `gennet/configs/model_config.py`

**Key Classes:**
- `ModernBERTConfig`: Configuration for text encoder
- `Siglip2Config`: Configuration for vision encoder
- `FusionConfig`: Configuration for fusion layer
- `RLConfig`: Configuration for RL layer
- `ModelConfig`: Main configuration that combines all sub-configs

**Example Configuration:**
```python
config = ModelConfig(
    text_config=ModernBERTConfig(
        model_name="answerdotai/ModernBERT-base",
        use_cls_token=True,
        finetune=True,
        use_domain_adaptive_pretraining=True,
        dap_special_tokens=["<DOMAIN_A>", "<DOMAIN_B>"]
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

## Usage

### Basic Training

```python
from gennet import MultiModalModel, MultiModalTrainer
from gennet.configs.model_config import ModelConfig

# Create model
config = ModelConfig(num_classes=10)
model = MultiModalModel(config)

# Create trainer
trainer = MultiModalTrainer(model, config)
trainer.configure_optimizers(learning_rate=1e-4)

# Train
trainer.train_epoch(train_loader, use_rl_loss=True)

# Evaluate
metrics = trainer.evaluate(val_loader)
```

### Inference

```python
import torch
from PIL import Image

# Load image
image = Image.open("image.jpg")

# Tokenize text (using model's tokenizer)
text = "A description of the image"

# Forward pass
outputs = model(
    text=text,
    images=image,
    return_dict=True
)

# Get results
action = outputs['rl_outputs']['action']
classification = outputs['classification_logits']
```

## Data Format

### Training Batch Format

```python
batch = {
    'text_input_ids': torch.Tensor,      # [batch_size, seq_length]
    'text_attention_mask': torch.Tensor, # [batch_size, seq_length]
    'pixel_values': torch.Tensor,        # [batch_size, 3, height, width]
    'labels': torch.Tensor,              # [batch_size] (optional)
    'rewards': torch.Tensor,             # [batch_size, 1] (optional for RL)
}
```

## Requirements

The implementation requires:
- PyTorch >= 2.0.0
- Transformers >= 4.35.0
- PIL (Pillow) >= 10.0.0
- NumPy >= 1.24.0
- tqdm >= 4.65.0

## Implementation Notes

### 1. ModernBERT Integration

The ModernBERT encoder is loaded from Hugging Face's transformers library using `AutoModel.from_pretrained()`. The implementation:
- Supports both `answerdotai/ModernBERT-base` and `answerdotai/ModernBERT-large`
- Can freeze/unfreeze parameters for fine-tuning
- Extracts CLS token and mean-pooled outputs
- Provides special token utilities for DAP (MLM)

### 2. DAP (Domain-Adaptive Pre-Training via MLM)

DAP is implemented as a training stage in the trainer that:
- Adds optional domain-specific special tokens
- Uses `AutoModelForMaskedLM` for masked language modeling
- Fine-tunes the text encoder on domain text
- Copies encoder weights back into the downstream text encoder

### 3. Siglip2 Integration

The Siglip2 vision encoder:
- Loads from `google/siglip-base-patch16-224` or similar models
- Can extract vision-only features without text encoder
- Supports both 224x224 and other image sizes
- Uses pre-trained weights from Google Research

### 4. Cross-Modal Fusion

The fusion layer:
- Projects features to a common dimension
- Uses bidirectional cross-attention
- Applies MLP transformation
- Includes residual connections for better gradient flow

### 5. RL Layer

The RL layer implements Actor-Critic:
- **Actor (Policy Network)**: Learns action distribution
- **Critic (Value Network)**: Estimates state values
- **Advantage**: Used to reduce variance in policy gradients
- **Action Embeddings**: Allow reasoning about selected actions

### 6. Training Strategy

The trainer supports staged training:
1. **Stage 1**: Train fusion and RL layers with frozen encoders
2. **Stage 2**: Unfreeze encoders and fine-tune end-to-end
3. **Stage 3**: Continue training with RL loss

## Testing

Run structure tests:
```bash
python tests/test_structure.py
```

Run example training (requires dependencies):
```bash
cd examples
python train_example.py
```

## File Organization

```
Gennet/
├── gennet/
│   ├── __init__.py
│   ├── configs/
│   │   ├── __init__.py
│   │   └── model_config.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── text_encoder.py
│   │   ├── vision_encoder.py
│   │   ├── fusion_layer.py
│   │   ├── rl_layer.py
│   │   └── multimodal_model.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils/
│       ├── __init__.py
│       └── data_utils.py
├── examples/
│   └── train_example.py
├── tests/
│   └── test_structure.py
├── requirements.txt
├── setup.py
└── README.md
```

## Future Enhancements

Potential improvements:
1. Add support for more vision encoders (CLIP, BLIP, etc.)
2. Implement more RL algorithms (PPO, A3C, etc.)
3. Add distributed training support
4. Implement curriculum learning for RL
5. Add more fusion strategies (gated fusion, transformer fusion, etc.)
6. Support for video inputs (temporal modeling)
7. Add pre-training objectives (contrastive learning, masked modeling, etc.)

## References

- ModernBERT: https://huggingface.co/answerdotai/ModernBERT-base
- Siglip2: https://huggingface.co/google/siglip-base-patch16-224
- PyTorch: https://pytorch.org/
- Hugging Face Transformers: https://huggingface.co/transformers/
