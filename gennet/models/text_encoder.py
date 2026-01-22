"""ModernBERT text encoder with DAP and CLS fine-tuning."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional


class DAPPooling(nn.Module):
    """Discriminative Adapter Pooling for ModernBERT."""
    
    def __init__(self, hidden_size: int, num_adapters: int = 4):
        super().__init__()
        self.num_adapters = num_adapters
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            ) for _ in range(num_adapters)
        ])
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply DAP to hidden states."""
        # Apply adapters
        adapted_states = []
        for adapter in self.adapters:
            adapted = adapter(hidden_states)
            adapted_states.append(adapted)
        
        # Stack and apply attention
        stacked = torch.stack(adapted_states, dim=1)  # [batch, num_adapters, seq_len, hidden]
        batch_size, num_adapters, seq_len, hidden_size = stacked.shape
        stacked = stacked.reshape(batch_size, num_adapters * seq_len, hidden_size)
        
        # Self-attention to pool
        pooled, _ = self.attention(stacked, stacked, stacked)
        pooled = pooled.mean(dim=1)  # [batch, hidden_size]
        
        return pooled


class ModernBERTEncoder(nn.Module):
    """ModernBERT encoder with CLS token and DAP fine-tuning."""
    
    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 512,
        use_cls_token: bool = True,
        use_dap: bool = True,
        finetune: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.use_cls_token = use_cls_token
        self.use_dap = use_dap
        self.finetune = finetune
        
        # Load ModernBERT model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Set requires_grad based on finetune flag
        if not finetune:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        
        # DAP pooling if enabled
        if use_dap:
            self.dap = DAPPooling(hidden_size)
        else:
            self.dap = None
            
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        text: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ModernBERT encoder.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            text: Raw text (will be tokenized if input_ids not provided)
            
        Returns:
            Dictionary with 'cls_output' and 'pooled_output'
        """
        # Tokenize if raw text provided
        if input_ids is None and text is not None:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract features
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        result = {}
        
        # CLS token output
        if self.use_cls_token:
            result['cls_output'] = last_hidden_state[:, 0, :]  # [batch, hidden]
        
        # DAP pooling
        if self.use_dap and self.dap is not None:
            result['pooled_output'] = self.dap(last_hidden_state)
        else:
            # Mean pooling as fallback
            result['pooled_output'] = last_hidden_state.mean(dim=1)
            
        return result
