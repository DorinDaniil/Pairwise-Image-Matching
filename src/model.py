import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import EfficientNetB3Encoder, CLIPEncoder


class HeadModel(nn.Module):
    """Head for image similarity prediction that accepts pre-computed difference embeddings.
    
    This architecture is designed to work with pre-computed difference vectors between embeddings.
    The input should already be in the format [|emb1 - emb2|] for all pairs.
    
    Note: For training, use BCEWithLogitsLoss instead of manually applying sigmoid.
    """
    
    def __init__(self, feature_dim, dropout_rate=0.1):
        """
        Args:
            feature_dim (int): Dimension of input feature vector (|emb1 - emb2|)
            dropout_rate (float): Dropout probability for regularization. Defaults to 0.1.
        """
        super().__init__()
        
        # Main prediction pathway - same dimension architecture
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 1)
        )
    
    def forward(self, diff_embeddings):
        """
        Forward pass for similarity prediction.
        
        Args:
            diff_embeddings: Tensor of shape [batch_size, feature_dim]
                representing pre-computed |embedding1 - embedding2|
        
        Returns:
            Tensor of shape [batch_size, 1] with raw logits (before sigmoid)
        """
        return self.predictor(diff_embeddings)


class AdvancedHeadModel(nn.Module):
    """Advanced head that accepts pre-computed difference AND product embeddings.
    
    This architecture is designed to work with pre-computed difference vectors AND element-wise products.
    The inputs should already be in the format:
    - diff_embeddings: [|emb1 - emb2|] for all pairs
    - product_embeddings: [emb1 * emb2] for all pairs
    
    Note: For training, use BCEWithLogitsLoss instead of manually applying sigmoid.
    """
    
    def __init__(self, feature_dim, dropout_rate=0.1):
        """
        Args:
            feature_dim (int): Dimension of input feature vectors
            dropout_rate (float): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        
        # Pathway for processing the absolute difference
        self.diff_pathway = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Pathway for processing element-wise product
        self.product_pathway = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Final fusion and prediction
        self.fusion = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, diff_embeddings, product_embeddings):
        """
        Forward pass with pre-computed difference and product features.
        
        Args:
            diff_embeddings: Tensor of shape [batch_size, feature_dim] 
                representing pre-computed |embedding1 - embedding2|
            product_embeddings: Tensor of shape [batch_size, feature_dim]
                representing pre-computed element-wise product of embeddings
                
        Returns:
            Tensor of shape [batch_size, 1] with raw logits
        """
        diff_features = self.diff_pathway(diff_embeddings)
        product_features = self.product_pathway(product_embeddings)
        
        # Concatenate and fuse both representations
        combined = torch.cat([diff_features, product_features], dim=1)
        return self.fusion(combined)


class SiamNet(nn.Module):
    """
    Siamese network with configurable encoder backbone.
    
    Args:
        model_name (str): Type of encoder to use ('efficientnet' or 'clip')
        freeze_encoder (bool): Whether to freeze encoder weights during training
        use_advanced_head (bool): Whether to use the advanced head with product features
    """
    def __init__(self, model_name='efficientnet', freeze_encoder=True, use_advanced_head=True):
        super(SiamNet, self).__init__()
        
        # Initialize the appropriate encoder based on model_name
        if model_name.lower() == 'efficientnet':
            self.encoder = EfficientNetB3Encoder(
                freeze=freeze_encoder
            )
        elif model_name.lower() == 'clip':
            self.encoder = CLIPEncoder(
                freeze=freeze_encoder
            )
        else:
            raise ValueError(f"Unsupported model name: {model_name}. Choose 'efficientnet' or 'clip'.")
        
        self.use_advanced_head = use_advanced_head
        if use_advanced_head:
            self.head = AdvancedHeadModel(self.encoder.feature_dim)
        else:
            self.head = HeadModel(self.encoder.feature_dim)

    def forward(self, batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the siamese network.
        
        Args:
            batch1: First batch of images [batch_size, channels, height, width]
            batch2: Second batch of images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Similarity matrix [batch_size1, batch_size2]
        """
        emb1 = self.encoder(batch1)
        emb2 = self.encoder(batch2)

        batch_size1, batch_size2 = emb1.size(0), emb2.size(0)
        emb1_expanded = emb1.unsqueeze(1).expand(-1, batch_size2, -1)
        emb2_expanded = emb2.unsqueeze(0).expand(batch_size1, -1, -1)

        # Pre-compute the difference and product vectors
        diff_embeddings = torch.abs(emb1_expanded - emb2_expanded)
        product_embeddings = emb1_expanded * emb2_expanded
        
        # Reshape for head model
        diff_embeddings = diff_embeddings.view(-1, self.encoder.feature_dim)
        product_embeddings = product_embeddings.view(-1, self.encoder.feature_dim)

        # Pass pre-computed features to the head
        if self.use_advanced_head:
            logits = self.head(diff_embeddings, product_embeddings)
        else:
            logits = self.head(diff_embeddings)
            
        logits = logits.view(batch_size1, batch_size2)
        return logits

    def predict_from_diff_and_product(self, diff_embeddings, product_embeddings):
        """
        Directly predict similarity from pre-computed difference and product vectors.
        
        Args:
            diff_embeddings: Tensor of shape [N, feature_dim] containing pre-computed |emb1 - emb2|
            product_embeddings: Tensor of shape [N, feature_dim] containing pre-computed emb1 * emb2
            
        Returns:
            torch.Tensor: Similarity scores [N, 1]
        """
        if not self.use_advanced_head:
            raise ValueError("This model was initialized with a simple head that doesn't use product embeddings")
        return self.head(diff_embeddings, product_embeddings)
    
    def predict_from_diff(self, diff_embeddings):
        """
        Directly predict similarity from pre-computed difference vectors.
        
        Args:
            diff_embeddings: Tensor of shape [N, feature_dim] containing pre-computed |emb1 - emb2|
            
        Returns:
            torch.Tensor: Similarity scores [N, 1]
        """
        if self.use_advanced_head:
            raise ValueError("This model was initialized with an advanced head that requires product embeddings")
        return self.head(diff_embeddings)
    
    def get_preprocessing(self):
        """Returns the preprocessing pipeline for the encoder"""
        return self.encoder.preprocess


def get_siamnet(config):
    device = config['training']['device']
    model_name = config['model'].get('model_name', 'efficientnet')
    net = SiamNet(model_name)

    if config['model']['weights_path']:
        net.load_state_dict(torch.load(config['model']['weights_path']))
        print(f"Loaded weights from {config['model']['weights_path']}")

    if config['model']['reinitialize_fc_layers']:
        net.head.fc1 = nn.Linear(net.head.fc1.in_features, net.head.fc1.out_features)
        net.head.fc2 = nn.Linear(net.head.fc2.in_features, net.head.fc2.out_features)
        print('Fully connected layers reinitialized')

    if config['model']['freeze_extractor_layers']:
        for param in net.encoder.parameters():
            param.requires_grad = False
        for param in net.head.parameters():
            param.requires_grad = True
        print('Feature extractor layers frozen')

    net = net.to(device)
    return net