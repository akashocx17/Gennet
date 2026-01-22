"""Trainer for multi-modal neural network with RL."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Any
from tqdm import tqdm
import logging

from gennet.models.multimodal_model import MultiModalModel
from gennet.configs.model_config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalTrainer:
    """
    Trainer for multi-modal model with reinforcement learning.
    
    Supports:
        - Supervised learning with classification loss
        - Reinforcement learning with policy gradient
        - Multi-modal fusion training
        - Staged training (freeze/unfreeze encoders)
    """
    
    def __init__(
        self,
        model: MultiModalModel,
        config: Optional[ModelConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.config = config if config is not None else ModelConfig()
        self.device = device
        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        
        # Optimizers (will be set in configure_optimizers)
        self.optimizer = None
        self.rl_optimizer = None
        
    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        rl_learning_rate: Optional[float] = None,
        weight_decay: float = 0.01
    ):
        """
        Configure optimizers for training.
        
        Args:
            learning_rate: Learning rate for main model
            rl_learning_rate: Learning rate for RL layer (uses learning_rate if None)
            weight_decay: Weight decay for regularization
        """
        if rl_learning_rate is None:
            rl_learning_rate = self.config.rl_config.learning_rate
        
        # Main optimizer for all parameters
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Separate optimizer for RL layer if needed
        if self.model.rl_layer is not None:
            self.rl_optimizer = torch.optim.AdamW(
                self.model.rl_layer.parameters(),
                lr=rl_learning_rate,
                weight_decay=weight_decay
            )
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        use_rl_loss: bool = True,
        classification_weight: float = 1.0,
        rl_policy_weight: float = 0.1,
        rl_value_weight: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with optional RL components.
        
        Args:
            outputs: Model outputs dictionary
            labels: Classification labels
            rewards: RL rewards
            use_rl_loss: Whether to include RL loss
            classification_weight: Weight for classification loss
            rl_policy_weight: Weight for RL policy loss
            rl_value_weight: Weight for RL value loss
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0
        
        # Classification loss
        if labels is not None and outputs.get('classification_logits') is not None:
            cls_loss = self.classification_criterion(
                outputs['classification_logits'],
                labels
            )
            losses['classification_loss'] = cls_loss
            total_loss += classification_weight * cls_loss
        
        # RL losses
        if use_rl_loss and 'rl_outputs' in outputs:
            rl_outputs = outputs['rl_outputs']
            
            # Policy gradient loss
            if 'log_prob' in rl_outputs and rewards is not None:
                # Compute returns from rewards
                returns = self.model.rl_layer.compute_returns(rewards)
                
                # Get values if available
                values = rl_outputs.get('state_value')
                
                # Compute policy loss
                policy_loss = self.model.rl_layer.compute_policy_loss(
                    log_probs=rl_outputs['log_prob'],
                    returns=returns,
                    values=values
                )
                losses['rl_policy_loss'] = policy_loss
                total_loss += rl_policy_weight * policy_loss
                
                # Value loss
                if values is not None:
                    value_loss = self.model.rl_layer.compute_value_loss(
                        values=values,
                        returns=returns
                    )
                    losses['rl_value_loss'] = value_loss
                    total_loss += rl_value_weight * value_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        use_rl_loss: bool = True
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of data
            use_rl_loss: Whether to use RL loss
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            text_input_ids=batch.get('text_input_ids'),
            text_attention_mask=batch.get('text_attention_mask'),
            pixel_values=batch.get('pixel_values'),
            return_rl_distribution=True,
            return_dict=True
        )
        
        # Compute loss
        losses = self.compute_loss(
            outputs=outputs,
            labels=batch.get('labels'),
            rewards=batch.get('rewards'),
            use_rl_loss=use_rl_loss
        )
        
        # Backward pass
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        if self.rl_optimizer is not None and use_rl_loss:
            self.rl_optimizer.zero_grad()
            
        losses['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        if self.optimizer is not None:
            self.optimizer.step()
        if self.rl_optimizer is not None and use_rl_loss:
            self.rl_optimizer.step()
        
        # Return loss values
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in losses.items()}
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        use_rl_loss: bool = True,
        log_interval: int = 10
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            use_rl_loss: Whether to use RL loss
            log_interval: Logging interval
            
        Returns:
            Average losses for the epoch
        """
        epoch_losses = {}
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            losses = self.train_step(batch, use_rl_loss=use_rl_loss)
            
            # Accumulate losses
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v
            
            num_batches += 1
            
            # Update progress bar
            if batch_idx % log_interval == 0:
                progress_bar.set_postfix({
                    k: f"{v:.4f}" for k, v in losses.items()
                })
        
        # Average losses
        epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        return epoch_losses
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        metric_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Evaluate model.
        
        Args:
            dataloader: Evaluation dataloader
            metric_fn: Optional metric function
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                text_input_ids=batch.get('text_input_ids'),
                text_attention_mask=batch.get('text_attention_mask'),
                pixel_values=batch.get('pixel_values'),
                return_rl_distribution=False,
                return_dict=True
            )
            
            # Compute loss if labels available
            if batch.get('labels') is not None:
                losses = self.compute_loss(
                    outputs=outputs,
                    labels=batch['labels'],
                    use_rl_loss=False
                )
                total_loss += losses['total_loss'].item()
                num_batches += 1
                
                # Collect predictions
                if outputs.get('classification_logits') is not None:
                    preds = torch.argmax(outputs['classification_logits'], dim=-1)
                    all_predictions.extend(preds.cpu().tolist())
                    all_labels.extend(batch['labels'].cpu().tolist())
        
        metrics = {}
        if num_batches > 0:
            metrics['loss'] = total_loss / num_batches
        
        # Compute custom metrics if provided
        if metric_fn is not None and len(all_predictions) > 0:
            custom_metrics = metric_fn(all_predictions, all_labels)
            metrics.update(custom_metrics)
        
        return metrics
    
    def save_checkpoint(self, path: str, epoch: int, **kwargs):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }
        
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.rl_optimizer is not None:
            checkpoint['rl_optimizer_state_dict'] = self.rl_optimizer.state_dict()
            
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.rl_optimizer is not None and 'rl_optimizer_state_dict' in checkpoint:
            self.rl_optimizer.load_state_dict(checkpoint['rl_optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint
