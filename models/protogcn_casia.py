"""ProtoGCN-derived CASIA adaptation.
Original reference: https://github.com/firework8/ProtoGCN/
"""

from __future__ import annotations

import torch
import torch.nn as nn

from datasets.graph import Graph


class TemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor):
        super().__init__()
        self.register_buffer('adjacency', adjacency)
        self.adaptive_adj = nn.Parameter(adjacency.clone())
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, T, V]
        support = torch.einsum('nctv,vw->nctw', x, self.adaptive_adj[0])
        out = self.proj(support)
        return self.act(self.bn(out))


class ProtoGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor, stride: int = 1):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, adjacency)
        self.tcn = TemporalConv(out_channels, out_channels, stride=stride)
        self.residual = nn.Identity() if (in_channels == out_channels and stride == 1) else TemporalConv(in_channels, out_channels, kernel_size=1, stride=stride)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.act(x + res)


class ProtoGCNCASIA(nn.Module):
    def __init__(self, num_classes: int, num_person: int = 1):
        super().__init__()
        graph = Graph('coco')
        adjacency = torch.tensor(graph.A, dtype=torch.float32)
        self.num_person = num_person
        self.num_classes = num_classes

        self.data_bn = nn.BatchNorm1d(3 * adjacency.size(-1))

        self.blocks = nn.Sequential(
            ProtoGCNBlock(3, 64, adjacency, stride=1),
            ProtoGCNBlock(64, 128, adjacency, stride=2),
            ProtoGCNBlock(128, 256, adjacency, stride=2),
            ProtoGCNBlock(256, 384, adjacency, stride=2),
        )

        self.proto = nn.Sequential(
            nn.Linear(384, 100, bias=False),
            nn.Softmax(dim=-1),
            nn.Linear(100, 384, bias=False),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(384, num_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        if x.ndim != 5:
            raise ValueError(f'Expected [N, T, V, C] or [N, M, T, V, C], got {tuple(x.shape)}')

        n, m, t, v, c = x.shape
        x = x.permute(0, 1, 3, 4, 2).contiguous().view(n * m, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
        return x

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None, return_embedding: bool = False):
        x = self._normalize_input(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        embedding = self.proto(x)
        logits = self.classifier(embedding)

        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            if return_embedding:
                return {'loss_cls': loss}, logits, embedding
            return {'loss_cls': loss}

        if return_embedding:
            return logits, embedding
        return logits