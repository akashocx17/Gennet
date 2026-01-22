"""Example training script for multi-modal model."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from gennet.models.multimodal_model import MultiModalModel
from gennet.training.trainer import MultiModalTrainer
from gennet.configs.model_config import (
    ModelConfig,
    ModernBERTConfig,
    Siglip2Config,
    FusionConfig,
    RLConfig
)
from gennet.utils.data_utils import create_dummy_batch


def create_dummy_dataset(num_samples: int = 100, num_classes: int = 10):
    """Create a dummy dataset for demonstration."""
    # Create dummy data
    text_input_ids = torch.randint(0, 30000, (num_samples, 128))
    text_attention_mask = torch.ones(num_samples, 128)
    pixel_values = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))
    rewards = torch.randn(num_samples, 1)
    
    dataset = TensorDataset(
        text_input_ids,
        text_attention_mask,
        pixel_values,
        labels,
        rewards
    )
    
    return dataset


def collate_fn(batch):
    """Custom collate function."""
    text_input_ids = torch.stack([item[0] for item in batch])
    text_attention_mask = torch.stack([item[1] for item in batch])
    pixel_values = torch.stack([item[2] for item in batch])
    labels = torch.stack([item[3] for item in batch])
    rewards = torch.stack([item[4] for item in batch])
    
    return {
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'pixel_values': pixel_values,
        'labels': labels,
        'rewards': rewards
    }


def main():
    """Main training script."""
    print("=" * 60)
    print("Multi-Modal Neural Network Training Example")
    print("=" * 60)
    
    # Configuration
    config = ModelConfig(
        text_config=ModernBERTConfig(
            model_name="answerdotai/ModernBERT-base",
            max_length=128,
            use_cls_token=True,
            use_dap=True,
            finetune=True
        ),
        vision_config=Siglip2Config(
            model_name="google/siglip-base-patch16-224",
            image_size=224,
            use_vision_only=True
        ),
        fusion_config=FusionConfig(
            fusion_hidden_size=1024,
            num_fusion_layers=2,
            dropout=0.1,
            activation="gelu"
        ),
        rl_config=RLConfig(
            num_actions=10,
            gamma=0.99,
            learning_rate=1e-4,
            use_policy_gradient=True,
            use_value_network=True
        ),
        num_classes=10
    )
    
    # Create model
    print("\n[1/5] Creating multi-modal model...")
    print(f"  - Text Encoder: {config.text_config.model_name}")
    print(f"  - Vision Encoder: {config.vision_config.model_name}")
    print(f"  - Fusion Layer: {config.fusion_config.num_fusion_layers} layers")
    print(f"  - RL Layer: {config.rl_config.num_actions} actions")
    
    model = MultiModalModel(config)
    
    # Print model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Create dummy dataset
    print("\n[2/5] Creating dummy dataset...")
    train_dataset = create_dummy_dataset(num_samples=100, num_classes=10)
    val_dataset = create_dummy_dataset(num_samples=20, num_classes=10)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    
    # Create trainer
    print("\n[3/5] Initializing trainer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  - Device: {device}")
    
    trainer = MultiModalTrainer(model, config, device=device)
    trainer.configure_optimizers(learning_rate=1e-4)
    
    # Training
    print("\n[4/5] Starting training...")
    num_epochs = 2
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_losses = trainer.train_epoch(train_loader, use_rl_loss=True)
        print(f"Training - {train_losses}")
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        print(f"Validation - {val_metrics}")
        
        # Save checkpoint
        checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pt"
        trainer.save_checkpoint(checkpoint_path, epoch + 1)
    
    print("\n[5/5] Training completed!")
    print("=" * 60)
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        dummy_batch = create_dummy_batch(batch_size=2, device=device)
        outputs = model(
            text_input_ids=dummy_batch['text_input_ids'],
            text_attention_mask=dummy_batch['text_attention_mask'],
            pixel_values=dummy_batch['pixel_values'],
            return_dict=True
        )
        
        print(f"  - Classification logits shape: {outputs['classification_logits'].shape}")
        print(f"  - Fused features shape: {outputs['fused_features'].shape}")
        print(f"  - RL action: {outputs['rl_outputs']['action']}")
        
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
