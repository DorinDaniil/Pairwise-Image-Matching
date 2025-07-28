import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import open_clip
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, model_name='efficientnet'):
        super(Encoder, self).__init__()
        self.model_name = model_name
        self.extractor, self.feature_dim = self._initialize_extractor()

    def _initialize_extractor(self):
        if self.model_name == 'efficientnet':
            extractor = EfficientNet.from_pretrained('efficientnet-b3')
            extractor = nn.Sequential(
                extractor._conv_stem,
                extractor._bn0,
                *extractor._blocks,
                extractor._conv_head,
                extractor._bn1,
                extractor._avg_pooling
            )
            feature_dim = 1536
        elif self.model_name == 'clip':
            model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
            extractor = model.visual
            feature_dim = 512
        elif self.model_name == 'barlow_twins':
            extractor = models.resnet50(pretrained=False)
            extractor.fc = nn.Identity()
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        return extractor, feature_dim

    def forward(self, x):
        if self.model_name == 'efficientnet':
            x = self.extractor.extract_features(x)
        elif self.model_name == 'clip':
            x = self.extractor(x)
        elif self.model_name == 'barlow_twins':
            pass
        return x