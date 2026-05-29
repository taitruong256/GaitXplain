import torch
import torch.nn as nn


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, padding=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                            stride=(stride, 1), padding=(padding, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class GaitXplain(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, num_classes=125, 
                 num_layers=3, num_nodes=17, dropout=0.5, **kwargs):
        super().__init__()
        self.st_blocks = nn.ModuleList()
        in_ch = in_channels
        for _ in range(num_layers):
            self.st_blocks.append(nn.Sequential(
                TemporalConv(in_ch, hidden_channels),
                nn.Dropout(dropout)
            ))
            in_ch = hidden_channels
        
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_channels // 2, num_classes)

    def forward_features(self, x):
        B, T, V, C = x.shape
        x = x.permute(0, 3, 1, 2)
        for block in self.st_blocks:
            x = block(x)
        x = x.mean(dim=(2, 3))
        embedding = torch.relu(self.fc1(x))
        embedding = self.dropout(embedding)
        logits = self.fc2(embedding)
        return logits, embedding

    def forward(self, x, **kwargs):
        logits, embedding = self.forward_features(x)
        if kwargs.get('return_embedding', False):
            return logits, embedding
        return logits
