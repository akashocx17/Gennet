"""Main multi-modal model integrating text, vision, fusion, and RL components."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from PIL import Image

from gennet.configs.model_config import ModelConfig
from gennet.models.text_encoder import ModernBERTEncoder
from gennet.models.vision_encoder import Siglip2VisionEncoder
from gennet.models.fusion_layer import CrossModalFusion
from gennet.models.rl_layer import ReinforcementLearningLayer


class MultiModalModel(nn.Module):
    """
    Multi-modal neural network combining ModernBERT (text) and Siglip2 (vision)
    with cross-modal fusion and reinforcement learning for reasoning.
    
    Architecture:
        1. Text Encoder: ModernBERT with DAP and CLS fine-tuning
        2. Vision Encoder: Siglip2 vision encoder
        3. Fusion Layer: MLP-based cross-modal fusion
        4. RL Layer: Actor-Critic for reasoning and decision making
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        
        # Initialize config
        if config is None:
            config = ModelConfig()
        self.config = config
        
        # Text encoder (ModernBERT)
        self.text_encoder = ModernBERTEncoder(
            model_name=config.text_config.model_name,
            max_length=config.text_config.max_length,
            use_cls_token=config.text_config.use_cls_token,
            use_dap=config.text_config.use_dap,
            finetune=config.text_config.finetune
        )
        
        # Vision encoder (Siglip2)
        self.vision_encoder = Siglip2VisionEncoder(
            model_name=config.vision_config.model_name,
            image_size=config.vision_config.image_size,
            use_vision_only=config.vision_config.use_vision_only
        )
        
        # Get hidden sizes
        text_hidden_size = config.text_config.hidden_size
        vision_hidden_size = config.vision_config.hidden_size
        
        # Cross-modal fusion layer
        self.fusion_layer = CrossModalFusion(
            text_hidden_size=text_hidden_size,
            vision_hidden_size=vision_hidden_size,
            fusion_hidden_size=config.fusion_config.fusion_hidden_size,
            num_layers=config.fusion_config.num_fusion_layers,
            dropout=config.fusion_config.dropout,
            activation=config.fusion_config.activation
        )
        
        # Reinforcement learning layer
        self.rl_layer = ReinforcementLearningLayer(
            input_size=config.fusion_config.fusion_hidden_size,
            num_actions=config.rl_config.num_actions,
            gamma=config.rl_config.gamma,
            use_policy_gradient=config.rl_config.use_policy_gradient,
            use_value_network=config.rl_config.use_value_network
        )
        
        # Optional classification head
        if config.num_classes is not None:
            self.classifier = nn.Linear(
                config.fusion_config.fusion_hidden_size,
                config.num_classes
            )
        else:
            self.classifier = None
            
    def forward(
        self,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        pixel_values: Optional[torch.Tensor] = None,
        images: Optional[Union[Image.Image, list]] = None,
        return_rl_distribution: bool = True,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass through the multi-modal model.
        
        Args:
            text_input_ids: Tokenized text input IDs
            text_attention_mask: Text attention mask
            text: Raw text string (alternative to input_ids)
            pixel_values: Preprocessed image pixel values
            images: PIL Image or list of images (alternative to pixel_values)
            return_rl_distribution: Whether to sample from RL action distribution
            return_dict: Whether to return dictionary or just classification logits
            
        Returns:
            Dictionary containing all intermediate outputs and final predictions,
            or just classification logits if return_dict=False
        """
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            text=text
        )
        
        # Use pooled output from text encoder
        text_features = text_outputs['pooled_output']
        
        # Encode vision
        vision_outputs = self.vision_encoder(
            pixel_values=pixel_values,
            images=images
        )
        
        # Use pooled output from vision encoder
        vision_features = vision_outputs['pooled_output']
        
        # Fuse modalities
        fusion_outputs = self.fusion_layer(
            text_features=text_features,
            vision_features=vision_features
        )
        
        fused_features = fusion_outputs['fused_output']
        
        # Apply RL layer for reasoning
        rl_outputs = self.rl_layer(
            state=fused_features,
            return_distribution=return_rl_distribution
        )
        
        # Get reasoned state (fused features + action reasoning)
        if 'reasoned_state' in rl_outputs:
            final_features = rl_outputs['reasoned_state']
        else:
            final_features = fused_features
        
        # Classification if classifier exists
        if self.classifier is not None:
            classification_logits = self.classifier(final_features)
        else:
            classification_logits = None
        
        if not return_dict:
            return classification_logits if classification_logits is not None else final_features
        
        # Return all outputs
        return {
            'text_features': text_features,
            'vision_features': vision_features,
            'fused_features': fused_features,
            'final_features': final_features,
            'classification_logits': classification_logits,
            'rl_outputs': rl_outputs,
            'fusion_outputs': fusion_outputs,
            'text_outputs': text_outputs,
            'vision_outputs': vision_outputs
        }
    
    def get_text_encoder(self) -> ModernBERTEncoder:
        """Get the text encoder."""
        return self.text_encoder
    
    def get_vision_encoder(self) -> Siglip2VisionEncoder:
        """Get the vision encoder."""
        return self.vision_encoder
    
    def get_fusion_layer(self) -> CrossModalFusion:
        """Get the fusion layer."""
        return self.fusion_layer
    
    def get_rl_layer(self) -> ReinforcementLearningLayer:
        """Get the RL layer."""
        return self.rl_layer
    
    def freeze_encoders(self):
        """Freeze text and vision encoders."""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_encoders(self):
        """Unfreeze text and vision encoders."""
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
