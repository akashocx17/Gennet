"""Siglip2 vision encoder using only vision encoder."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from typing import Dict, Optional, Union
from PIL import Image


class Siglip2VisionEncoder(nn.Module):
    """Siglip2 vision encoder for image processing."""
    
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        image_size: int = 224,
        use_vision_only: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        self.use_vision_only = use_vision_only
        
        # Load Siglip2 model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Extract vision encoder if using vision only
        if use_vision_only:
            self.vision_encoder = self.model.vision_model
            self.hidden_size = self.vision_encoder.config.hidden_size
        else:
            self.vision_encoder = self.model
            self.hidden_size = self.model.config.vision_config.hidden_size
            
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        images: Optional[Union[Image.Image, list]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Siglip2 vision encoder.
        
        Args:
            pixel_values: Preprocessed pixel values
            images: PIL Image or list of images (will be processed if pixel_values not provided)
            
        Returns:
            Dictionary with 'pooled_output' and 'last_hidden_state'
        """
        # Process images if raw images provided
        if pixel_values is None and images is not None:
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs['pixel_values']
            
        # Move to same device as model
        if pixel_values is not None:
            pixel_values = pixel_values.to(next(self.vision_encoder.parameters()).device)
            
        # Get vision outputs
        if self.use_vision_only:
            outputs = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        else:
            outputs = self.vision_encoder.get_image_features(pixel_values=pixel_values, return_dict=True)
        
        result = {
            'pooled_output': outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1),
            'last_hidden_state': outputs.last_hidden_state
        }
        
        return result
