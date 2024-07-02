import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from data_vis import SpineDataset

# Access to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoubleConv(nn.Module):  # same as above.
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MultiLevelUnet(nn.Module):
    def __init__(self, in_channels=1, num_heads=25, num_classes_per_head=3, features=[64, 128, 256, 512]):
        super(MultiLevelUnet, self).__init__()
        self.num_heads = num_heads
        self.num_classes_per_head = num_classes_per_head
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # Constructing the down part of U-Net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Classifier for multi-level features
        # Including bottleneck features
        total_features = sum(features) + features[-1] * 2
        self.shared_fc = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512)
        )

        # Define the heads
        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes_per_head) for _ in range(num_heads)
        ])

    def forward(self, x):
        feature_maps = []

        # Downsample path
        for down in self.downs:
            x = down(x)
            feature_maps.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        feature_maps.append(x)

        # Concatenate all feature maps
        global_features = [F.adaptive_avg_pool2d(feature, (1, 1)).view(
            x.size(0), -1) for feature in feature_maps]
        global_features = torch.cat(global_features, 1)

        # Shared fully connected layer
        x = self.shared_fc(global_features)

        # Collect outputs from each head
        outputs = [head(x) for head in self.heads]
        return outputs


model = MultiLevelUnet(in_channels=1, num_heads=25, num_classes_per_head=3)
# Example input tensor with batch size 4
input_tensor = torch.randn(4, 1, 256, 256)
outputs = model(input_tensor)

# Each output corresponds to the predictions for a specific condition-level combination
for i, output in enumerate(outputs):
    print(f"Output for head {i}: {output.shape}")

