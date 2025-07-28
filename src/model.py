import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import open_clip
import torchvision.models as models

class HeadModel(nn.Module):
    def __init__(self, feature_dim):
        super(HeadModel, self).__init__()
        self.dropf = nn.Dropout(p=0.20)
        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.bn1 = nn.BatchNorm1d(feature_dim)
        self.drop1 = nn.Dropout(p=0.20)
        self.fc2 = nn.Linear(feature_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, combined_embeddings):
        r = self.fc1(combined_embeddings)
        r = self.bn1(r)
        r = F.relu(r)
        r = self.drop1(r)
        r = self.fc2(r)
        r = self.sigmoid(r)
        return r

class SiamNet(nn.Module):
    def __init__(self, model_name='efficientnet'):
        super(SiamNet, self).__init__()
        self.encoder = Encoder(model_name)
        self.head = HeadModel(self.encoder.feature_dim)

    def forward(self, batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
        emb1 = self.encoder(batch1)
        emb2 = self.encoder(batch2)

        batch_size1, batch_size2 = emb1.size(0), emb2.size(0)
        emb1_expanded = emb1.unsqueeze(1).expand(-1, batch_size2, -1)
        emb2_expanded = emb2.unsqueeze(0).expand(batch_size1, -1, -1)

        combined_embeddings = torch.abs(emb1_expanded - emb2_expanded)
        combined_embeddings = combined_embeddings.view(-1, self.encoder.feature_dim)

        logits = self.head(combined_embeddings)
        logits = logits.view(batch_size1, batch_size2)
        return logits

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