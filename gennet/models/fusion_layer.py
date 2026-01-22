"""Cross-modal fusion layer for combining text and vision features."""

import torch
import torch.nn as nn
from typing import Dict, Optional


class CrossModalFusion(nn.Module):
    """Cross-attention based fusion layer for text and vision features."""
    
    def __init__(
        self,
        text_hidden_size: int = 768,
        vision_hidden_size: int = 768,
        fusion_hidden_size: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.text_hidden_size = text_hidden_size
        self.vision_hidden_size = vision_hidden_size
        self.fusion_hidden_size = fusion_hidden_size
        
        # Get activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Project text and vision to same dimension
        self.text_proj = nn.Linear(text_hidden_size, fusion_hidden_size)
        self.vision_proj = nn.Linear(vision_hidden_size, fusion_hidden_size)
        
        # Cross-attention for modality interaction
        self.cross_attention = nn.MultiheadAttention(
            fusion_hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion MLP layers
        fusion_layers = []
        for i in range(num_layers):
            if i == 0:
                # First layer takes concatenated features
                fusion_layers.extend([
                    nn.Linear(fusion_hidden_size * 2, fusion_hidden_size),
                    self.activation,
                    nn.Dropout(dropout)
                ])
            else:
                fusion_layers.extend([
                    nn.Linear(fusion_hidden_size, fusion_hidden_size),
                    self.activation,
                    nn.Dropout(dropout)
                ])
        
        self.fusion_mlp = nn.Sequential(*fusion_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(fusion_hidden_size)
        
    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse text and vision features.
        
        Args:
            text_features: Text features [batch, seq_len, text_hidden_size]
            vision_features: Vision features [batch, seq_len, vision_hidden_size]
            text_attention_mask: Mask for text [batch, seq_len] (1 for valid, 0 for pad)
            
        Returns:
            Dictionary with 'fused_output' and intermediate features
        """
        # Project to fusion dimension
        text_proj = self.text_proj(text_features)  # [batch, text_seq, fusion_hidden]
        vision_proj = self.vision_proj(vision_features)  # [batch, vis_seq, fusion_hidden]
        
        # Prepare padding mask for text (MultiheadAttention expects True for padding)
        key_padding_mask = None
        if text_attention_mask is not None:
            key_padding_mask = (text_attention_mask == 0)
        
        # Cross-attention: text attends to vision
        # Query: text, Key/Value: vision
        text_attended, _ = self.cross_attention(text_proj, vision_proj, vision_proj)
        
        # Cross-attention: vision attends to text
        # Query: vision, Key/Value: text
        # We need to pass key_padding_mask for text since it acts as Key here
        vision_attended, _ = self.cross_attention(vision_proj, text_proj, text_proj, key_padding_mask=key_padding_mask)
        
        # Pooling
        if text_attention_mask is not None:
            # Masked mean pooling for text
            mask = text_attention_mask.unsqueeze(-1).float()
            text_pooled = (text_attended * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            text_pooled = text_attended.mean(1)
            
        vision_pooled = vision_attended.mean(1)
        
        # Concatenate attended features
        concat_features = torch.cat([text_pooled, vision_pooled], dim=-1)
        
        # Apply fusion MLP
        fused = self.fusion_mlp(concat_features)
        
        # Residual connection and normalization (add pooled features)
        fused = self.layer_norm(fused + text_pooled + vision_pooled)
        
        return {
            'fused_output': fused,
            'text_proj': text_proj,
            'vision_proj': vision_proj,
            'text_attended': text_attended,
            'vision_attended': vision_attended
        }
