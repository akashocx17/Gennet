"""Cross-modal fusion layer for combining text and vision features."""

import torch
import torch.nn as nn
from typing import Dict


class CrossModalFusion(nn.Module):
    """MLP-based cross-modal fusion layer for text and vision features."""
    
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
        vision_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse text and vision features.
        
        Args:
            text_features: Text features [batch, text_hidden_size]
            vision_features: Vision features [batch, vision_hidden_size]
            
        Returns:
            Dictionary with 'fused_output' and intermediate features
        """
        # Project to fusion dimension
        text_proj = self.text_proj(text_features)  # [batch, fusion_hidden]
        vision_proj = self.vision_proj(vision_features)  # [batch, fusion_hidden]
        
        # Add sequence dimension for attention
        text_seq = text_proj.unsqueeze(1)  # [batch, 1, fusion_hidden]
        vision_seq = vision_proj.unsqueeze(1)  # [batch, 1, fusion_hidden]
        
        # Cross-attention: text attends to vision
        text_attended, _ = self.cross_attention(text_seq, vision_seq, vision_seq)
        text_attended = text_attended.squeeze(1)  # [batch, fusion_hidden]
        
        # Cross-attention: vision attends to text
        vision_attended, _ = self.cross_attention(vision_seq, text_seq, text_seq)
        vision_attended = vision_attended.squeeze(1)  # [batch, fusion_hidden]
        
        # Concatenate attended features
        concat_features = torch.cat([text_attended, vision_attended], dim=-1)
        
        # Apply fusion MLP
        fused = self.fusion_mlp(concat_features)
        
        # Residual connection and normalization
        fused = self.layer_norm(fused + text_proj + vision_proj)
        
        return {
            'fused_output': fused,
            'text_proj': text_proj,
            'vision_proj': vision_proj,
            'text_attended': text_attended,
            'vision_attended': vision_attended
        }
