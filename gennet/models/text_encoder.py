"""ModernBERT text encoder with CLS fine-tuning and utilities for DAP (Domain-Adaptive Pre-Training)."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, List


class ModernBERTEncoder(nn.Module):
    """ModernBERT encoder with CLS token fine-tuning.

    Provides utility methods to support Domain-Adaptive Pre-Training (DAP)
    outside the forward pass (e.g., special token handling).
    """
    
    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 512,
        use_cls_token: bool = True,
        finetune: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.use_cls_token = use_cls_token
        self.finetune = finetune
        
        # Load ModernBERT model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Set requires_grad based on finetune flag
        if not finetune:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.hidden_size = self.bert.config.hidden_size
            
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
            device = next(self.bert.parameters()).device
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
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
        
        # Mean pooling with attention mask
        if attention_mask is not None:
            # Expand mask: [batch, seq_len] -> [batch, seq_len, hidden]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            # Sum embeddings * mask
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            # Sum mask (clamp to avoid div by zero)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            result['pooled_output'] = sum_embeddings / sum_mask
        else:
            result['pooled_output'] = last_hidden_state.mean(dim=1)
            
        result['last_hidden_state'] = last_hidden_state
        result['attention_mask'] = attention_mask
        
        return result

    def add_special_tokens(self, special_tokens: Optional[List[str]] = None) -> int:
        """Add domain-specific special tokens and resize embeddings.

        Args:
            special_tokens: List of new special tokens to add.

        Returns:
            The number of tokens added.
        """
        if not special_tokens:
            return 0
        added = self.tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })
        if added > 0:
            self.bert.resize_token_embeddings(len(self.tokenizer))
        return added

    def get_tokenizer(self) -> AutoTokenizer:
        """Return the underlying tokenizer."""
        return self.tokenizer
