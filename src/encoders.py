import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import open_clip
from torchvision import transforms


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
        
        # Load pretrained model
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        
        # Replace dropout with Identity to remove it while preserving 1536 features
        self.backbone._dropout = nn.Identity()
        
        # Replace classifier with Identity to get 1536 features directly
        self.backbone._fc = nn.Identity()
        
        # Configure trainability
        self._set_trainable(freeze)
        
        # Fixed output dimension
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
    Pure CLIP ViT-B-32 encoder returning 512-dimensional features.
    
    Standard preprocessing is built-in as an attribute from open_clip:
    encoder.preprocess  # Contains proper CLIP preprocessing pipeline
    
    Example usage:
    # Apply preprocessing to PIL image
    image_tensor = encoder.preprocess(pil_image)
    
    # Get features
    features = encoder(image_tensor.unsqueeze(0))
    
    # Frozen encoder (for feature extraction)
    encoder = CLIPEncoder(freeze=True)
    
    # Unfreeze last 3 transformer blocks for fine-tuning
    encoder.unfreeze_last_layers(n_layers=3)
    """
    
    def __init__(self, freeze=True):
        """
        Args:
            freeze (bool): Freeze backbone weights (True by default)
        """
        super().__init__()
        
        # Load pretrained CLIP model AND its standard preprocessing
        model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        
        # Extract visual encoder
        self.backbone = model.visual
        
        # Configure trainability
        self._set_trainable(freeze)
        
        # Fixed output dimension
        self.feature_dim = 512

    def _set_trainable(self, freeze):
        """Configure layer trainability"""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = not freeze

    def unfreeze_last_layers(self, n_layers=3):
        """
        Unfreeze N last layers of the transformer backbone
        
        Args:
            n_layers (int): Number of last transformer layers to unfreeze
            
        CLIP ViT-B/32 structure:
        [patch embedding] -> [transformer blocks (12 blocks)] -> [ln_post] -> [proj]
        """
        # First freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last transformer blocks
        total_layers = len(self.backbone.transformer.resblocks)
        layers_to_unfreeze = min(n_layers, total_layers)
        
        for i in range(total_layers - layers_to_unfreeze, total_layers):
            for param in self.backbone.transformer.resblocks[i].parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        Forward pass through encoder
        
        Args:
            x: Input tensor [batch, 3, 224, 224] (after standard preprocessing)
            
        Returns:
            torch.Tensor: Feature embeddings [batch, 512]
        """
        return self.backbone(x)