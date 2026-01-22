# Getting Started with Gennet

This guide will help you get started with Gennet, a multi-modal neural network framework.

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/akashocx17/Gennet.git
cd Gennet
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

## Quick Start

### 1. Create a Model

```python
from gennet import MultiModalModel
from gennet.configs.model_config import ModelConfig

# Create configuration
config = ModelConfig(num_classes=10)

# Create model
model = MultiModalModel(config)
```

### 2. Prepare Data

Your data should include:
- Text inputs (tokenized)
- Image inputs (preprocessed)
- Labels (for supervised learning)
- Rewards (for RL, optional)

```python
import torch

# Example batch
batch = {
    'text_input_ids': torch.randint(0, 30000, (4, 128)),
    'text_attention_mask': torch.ones(4, 128),
    'pixel_values': torch.randn(4, 3, 224, 224),
    'labels': torch.randint(0, 10, (4,)),
    'rewards': torch.randn(4, 1)
}
```

### 3. Create Trainer

```python
from gennet import MultiModalTrainer

# Create trainer
trainer = MultiModalTrainer(model, config)

# Configure optimizers
trainer.configure_optimizers(learning_rate=1e-4)
```

### 4. Train

```python
from torch.utils.data import DataLoader

# Assuming you have train_dataset ready
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Train for one epoch
losses = trainer.train_epoch(train_loader, use_rl_loss=True)
print(f"Training losses: {losses}")
```

### 5. Evaluate

```python
# Assuming you have val_dataset ready
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Evaluate
metrics = trainer.evaluate(val_loader)
print(f"Validation metrics: {metrics}")
```

### 6. Save Model

```python
# Save checkpoint
trainer.save_checkpoint("my_model.pt", epoch=1)
```

## Running the Example

A complete example is provided in `examples/train_example.py`:

```bash
cd examples
python train_example.py
```

This will:
1. Create a multi-modal model
2. Generate dummy data
3. Train for 2 epochs
4. Evaluate on validation set
5. Save checkpoints
6. Test inference

## Customization

### Custom Text Encoder

```python
from gennet.configs.model_config import ModelConfig, ModernBERTConfig

config = ModelConfig(
    text_config=ModernBERTConfig(
        model_name="answerdotai/ModernBERT-large",  # Use large model
        max_length=512,
        finetune=True,
        use_domain_adaptive_pretraining=True,
        dap_special_tokens=["<DOMAIN_A>", "<DOMAIN_B>"],
        dap_mlm_epochs=1
    )
)

### Domain-Adaptive Pre-Training (MLM)

Run a brief MLM fine-tuning stage with domain-specific special tokens before supervised training:

```python
from torch.utils.data import DataLoader

domain_texts = [
    "Clinical findings indicate elevated markers <DOMAIN_A>.",
    "Financial report shows increased revenue <DOMAIN_B>."
]
text_loader = DataLoader(domain_texts, batch_size=2, shuffle=True)

trainer.domain_adaptive_pretrain_mlm(
    text_dataloader=text_loader,
    epochs=config.text_config.dap_mlm_epochs,
    special_tokens=config.text_config.dap_special_tokens
)
```
```

### Custom Vision Encoder

```python
from gennet.configs.model_config import Siglip2Config

config.vision_config = Siglip2Config(
    model_name="google/siglip-large-patch16-384",
    image_size=384  # Higher resolution
)
```

### Custom Fusion Layer

```python
from gennet.configs.model_config import FusionConfig

config.fusion_config = FusionConfig(
    fusion_hidden_size=2048,  # Larger fusion dimension
    num_fusion_layers=3,      # More layers
    dropout=0.2
)
```

### Custom RL Configuration

```python
from gennet.configs.model_config import RLConfig

config.rl_config = RLConfig(
    num_actions=20,          # More actions
    gamma=0.95,              # Different discount factor
    use_policy_gradient=True,
    use_value_network=True
)
```

## Inference

### Single Sample Inference

```python
import torch
from PIL import Image

# Load your model
model.eval()

# Prepare inputs
text = "A cat sitting on a table"
image = Image.open("cat.jpg")

# Inference
with torch.no_grad():
    outputs = model(
        text=text,
        images=image,
        return_dict=True
    )

# Get results
action = outputs['rl_outputs']['action']
probs = outputs['rl_outputs']['action_probs']
logits = outputs['classification_logits']

print(f"Selected action: {action}")
print(f"Action probabilities: {probs}")
print(f"Classification logits: {logits}")
```

### Batch Inference

```python
import torch
from torch.utils.data import DataLoader

model.eval()

predictions = []
with torch.no_grad():
    for batch in test_loader:
        outputs = model(
            text_input_ids=batch['text_input_ids'],
            text_attention_mask=batch['text_attention_mask'],
            pixel_values=batch['pixel_values'],
            return_dict=True
        )
        
        preds = torch.argmax(outputs['classification_logits'], dim=-1)
        predictions.extend(preds.cpu().tolist())
```

## Training Strategies

### Strategy 1: Supervised Only

Train with classification loss only:

```python
losses = trainer.train_epoch(train_loader, use_rl_loss=False)
```

### Strategy 2: RL Only

Train with RL losses only (requires rewards):

```python
losses = trainer.train_epoch(train_loader, use_rl_loss=True)
```

### Strategy 3: Staged Training

1. First freeze encoders and train fusion + RL:

```python
model.freeze_encoders()
for epoch in range(5):
    losses = trainer.train_epoch(train_loader, use_rl_loss=True)
```

2. Then unfreeze and fine-tune everything:

```python
model.unfreeze_encoders()
for epoch in range(10):
    losses = trainer.train_epoch(train_loader, use_rl_loss=True)
```

## Tips

1. **Start Small**: Begin with a small model and dataset to verify everything works
2. **Monitor Losses**: Check that losses are decreasing during training
3. **Use GPU**: Training will be much faster on GPU
4. **Save Checkpoints**: Save regularly to avoid losing progress
5. **Tune Hyperparameters**: Experiment with learning rates, batch sizes, etc.
6. **Validate Regularly**: Run evaluation to catch overfitting early

## Troubleshooting

### Out of Memory

- Reduce batch size
- Use gradient accumulation
- Freeze encoders
- Use smaller models

### Poor Performance

- Increase training epochs
- Tune learning rate
- Try different optimizers
- Check data quality
- Adjust model architecture

### Slow Training

- Use GPU if available
- Increase batch size
- Use mixed precision training
- Profile your code

## Next Steps

- Read the [IMPLEMENTATION.md](IMPLEMENTATION.md) for detailed architecture
- Check out [README.md](README.md) for full documentation
- Explore the code in `gennet/models/` to understand components
- Try different configurations and datasets

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing documentation
- Review the example code

Happy training! 🚀
