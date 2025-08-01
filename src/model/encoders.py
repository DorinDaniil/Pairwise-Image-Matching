import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import open_clip
from torchvision import transforms
from torchvision.models import vit_l_16, ViT_L_16_Weights


class EfficientNetB3Encoder(nn.Module):
    """
    Pure EfficientNet-B3 encoder returning 1536-dimensional features.
    
    Standard preprocessing is built-in as an attribute:
    encoder.preprocess  # Contains Resize(224) + ToTensor + Normalize
    
    Example usage:
    # Apply preprocessing to PIL image
    image_tensor = encoder.preprocess(pil_image)
    
    # Get features
    features = encoder(image_tensor.unsqueeze(0))
    
    # Frozen encoder (for feature extraction)
    encoder = EfficientNetB3Encoder(freeze=True)
    
    # Unfreeze last 4 layers for fine-tuning
    encoder.unfreeze_last_layers(n_layers=4)
    """
    def __init__(self, freeze=True):
        """
        Args:
            freeze (bool): Freeze backbone weights (True by default)
        """
        super().__init__()
        
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        self.backbone._dropout = nn.Identity()
        self.backbone._fc = nn.Identity()
    
        self._set_trainable(freeze)
        self.feature_dim = 1536
        
        # Built-in preprocessing pipeline (standard for EfficientNet)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _set_trainable(self, freeze):
        """Configure layer trainability"""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = not freeze

    def unfreeze_last_layers(self, n_layers=3):
        """
        Unfreeze N last layers of the backbone
        
        Args:
            n_layers (int): Number of last layers to unfreeze
            
        EfficientNet-B3 structure:
        [stem] -> [blocks (26 blocks)] -> [head (conv_head + bn1)] -> [pooling] -> [fc]
        """
        # First freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 1. Unfreeze final head (conv_head + bn1)
        for param in self.backbone._conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone._bn1.parameters():
            param.requires_grad = True
        n_layers -= 1  # Count head as 1 layer
        
        # 2. Unfreeze last MBConv blocks
        total_blocks = len(self.backbone._blocks)
        blocks_to_unfreeze = min(n_layers, total_blocks)
        
        for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
            for param in self.backbone._blocks[i].parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        Forward pass through encoder
        
        Args:
            x: Input tensor [batch, 3, 224, 224] (after standard preprocessing)
            
        Returns:
            torch.Tensor: Feature embeddings [batch, 1536]
        """
        return self.backbone(x)


class CLIPEncoder(nn.Module):
    """
    Pure CLIP ViT-H-14 encoder returning 1024-dimensional features.
    Uses LAION-2B pretrained weights with standard FP32 precision.
    
    Standard preprocessing is built-in as an attribute from open_clip:
    encoder.preprocess  # Contains proper CLIP preprocessing pipeline
    
    Example usage:
    # Apply preprocessing to PIL image
    image_tensor = encoder.preprocess(pil_image)
    
    # Get features
    features = encoder(image_tensor.unsqueeze(0))
    
    # Frozen encoder (for feature extraction)
    encoder = CLIPEncoder(freeze=True)
    
    # Unfreeze last 5 transformer blocks for fine-tuning
    encoder.unfreeze_last_layers(n_layers=5)
    """
    
    def __init__(self, freeze=True):
        """
        Args:
            freeze (bool): Freeze backbone weights (True by default)
        """
        super().__init__()
        
        # Load pretrained CLIP model with LAION-2B weights (FP32 by default)
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-H-14',
            pretrained='laion2b_s32b_b79k'
        )
        
        self.backbone = model.visual
        self._set_trainable(freeze)
        self.feature_dim = 1024

    def _set_trainable(self, freeze):
        """Configure layer trainability"""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = not freeze

    def unfreeze_last_layers(self, n_layers=5):
        """
        Unfreeze N last layers of the transformer backbone
        
        Args:
            n_layers (int): Number of last transformer layers to unfreeze
            
        ViT-H/14 structure:
        [patch embedding] -> [transformer blocks (32 blocks)] -> [ln_post] -> [proj]
        """
        # First freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 1. Unfreeze final layer norm (ln_post)
        for param in self.backbone.ln_post.parameters():
            param.requires_grad = True
        n_layers -= 1  # Count ln_post as 1 layer
        
        # 2. Unfreeze last transformer blocks (32 total blocks in ViT-H)
        total_blocks = len(self.backbone.transformer.resblocks)
        blocks_to_unfreeze = min(n_layers, total_blocks)
        
        for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
            for param in self.backbone.transformer.resblocks[i].parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        Forward pass through encoder
        
        Args:
            x: Input tensor [batch, 3, H, W] (after standard preprocessing)
            
        Returns:
            torch.Tensor: Feature embeddings [batch, 1024]
        """
        return self.backbone(x)
    

class ViTEncoder(nn.Module):
    """
    Pure Vision Transformer encoder using PyTorch's ViT-L/16 model with SWAG weights.
    Returns 1024-dimensional features matching ViT-L/16 implementation.
    
    Standard preprocessing is built-in as an attribute:
    encoder.preprocess  # Contains Resize(224) + CenterCrop(224) + Normalize
    
    Example usage:
    # Apply preprocessing to PIL image
    image_tensor = encoder.preprocess(pil_image)
    
    # Get features
    features = encoder(image_tensor.unsqueeze(0))
    
    # Frozen encoder (for feature extraction)
    encoder = ViTEncoder(freeze=True)
    
    # Unfreeze last 3 transformer blocks for fine-tuning
    encoder.unfreeze_last_layers(n_layers=3)
    """
    
    def __init__(self, freeze=True):
        """
        Args:
            freeze (bool): Freeze backbone weights (True by default)
        """
        super().__init__()
        
        # Load pretrained ViT-L/16 model
        weights = ViT_L_16_Weights.DEFAULT
        self.backbone = vit_l_16(weights=weights)
        self.backbone.heads = nn.Identity()
        self._set_trainable(freeze)
        self.feature_dim = 1024  # ViT-L/16 feature dimension
        self.preprocess = weights.transforms()

    def _set_trainable(self, freeze):
        """Configure layer trainability"""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def unfreeze_last_layers(self, n_layers=3):
        """
        Unfreeze N last layers of the transformer backbone
        
        Args:
            n_layers (int): Number of last transformer blocks to unfreeze
            
        ViT-L/16 structure:
        [conv_proj] -> [class_token, pos_embedding] -> 
        [transformer blocks (24 blocks)] -> [encoder_norm] -> [heads]
        """
        # First freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 1. Unfreeze final layer norm
        for param in self.backbone.encoder_norm.parameters():
            param.requires_grad = True
        n_layers -= 1  # Count norm as 1 layer
        
        # 2. Unfreeze last transformer blocks (24 total blocks in ViT-L/16)
        total_blocks = len(self.backbone.encoder.layers)
        assert total_blocks == 24, "ViT-L/16 should have 24 transformer blocks"
        blocks_to_unfreeze = min(n_layers, total_blocks)
        
        for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
            for param in self.backbone.encoder.layers[i].parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        Forward pass through encoder
        
        Args:
            x: Input tensor [batch, 3, H, W] (after standard preprocessing)
               Must be 224x224 as per ViT-L/16 requirements
            
        Returns:
            torch.Tensor: Feature embeddings [batch, 1024]
        """
        return self.backbone(x)