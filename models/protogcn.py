import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


class Graph:
    def __init__(self, dataset: str = "coco", max_hop: int = 3, dilation: int = 1):
        self.dataset = dataset.split("-")[0]
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge()
        self.A = self._get_adjacency()

    def _get_edge(self):
        if self.dataset == "coco":
            num_node = 17
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [
                (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
                (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16),
            ]
            connect_joint = np.array([0, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14])
            parts = [
                np.array([5, 7, 9]),
                np.array([6, 8, 10]),
                np.array([11, 13, 15]),
                np.array([12, 14, 16]),
                np.array([0, 1, 2, 3, 4]),
            ]
            edge = self_link + neighbor_link
            return num_node, edge, connect_joint, parts
        raise ValueError(f"Unsupported dataset: {self.dataset}")

    def _get_hop_distance(self):
        adjacency = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            adjacency[i, j] = 1
            adjacency[j, i] = 1
        hop_dis = np.ones((self.num_node, self.num_node)) * np.inf
        transfer_mat = [np.linalg.matrix_power(adjacency, d) for d in range(self.max_hop + 1)]
        arrive_mat = np.stack(transfer_mat) > 0
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        Dn = np.zeros((A.shape[0], A.shape[0]))
        for i in range(A.shape[0]):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        return np.dot(A, Dn)

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A


class UnitTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, dropout=0.0):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))


class MSTCN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.0,
        ms_cfg=((3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"),
        stride=1,
    ):
        super().__init__()
        self.act = nn.ReLU()
        num_branches = len(ms_cfg)
        mid_channels = out_channels // num_branches
        rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == "1x1":
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
            elif cfg[0] == "max":
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2d(branch_c),
                        self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0)),
                    )
                )
            else:
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2d(branch_c),
                        self.act,
                        UnitTCN(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1]),
                    )
                )
        self.branches = nn.ModuleList(branches)
        self.transform = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            self.act,
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        out = torch.cat(outs, dim=1)
        out = self.transform(out)
        return self.drop(self.bn(out))


class UnitGCN(nn.Module):
    def __init__(self, in_channels, out_channels, A, ratio=0.125):
        super().__init__()
        self.num_subsets = A.shape[0]
        self.mid_channels = max(1, int(ratio * out_channels))
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32), requires_grad=True)
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels * self.num_subsets, kernel_size=1),
            nn.BatchNorm2d(self.mid_channels * self.num_subsets),
            nn.ReLU(inplace=True),
        )
        self.post = nn.Conv2d(self.mid_channels * self.num_subsets, out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels * self.num_subsets, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, self.mid_channels * self.num_subsets, kernel_size=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-2)
        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))
        self.bn = nn.BatchNorm2d(out_channels)
        self.down = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        n, c, t, v = x.shape
        res = self.down(x)
        base_A = self.A[None, :, None, :, :]
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        x1 = self.conv1(x).reshape(n, self.num_subsets, self.mid_channels, t, v).mean(dim=-2, keepdim=True)
        x2 = self.conv2(x).reshape(n, self.num_subsets, self.mid_channels, t, v).mean(dim=-2, keepdim=True)

        diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
        # reduce channel dimension so inter_graph matches adjacency shape
        inter_graph = self.tanh(diff).mean(dim=2)
        # apply per-subset alpha (broadcast to match inter_graph dims)
        inter_graph = inter_graph * self.alpha.view(1, self.num_subsets, 1, 1, 1)

        intra_graph = torch.einsum("nkctv,nkctw->nktvw", x1, x2)
        # apply softmax and per-subset beta (broadcast across subset dim only)
        intra_graph = self.softmax(intra_graph) * self.beta.view(1, self.num_subsets, 1, 1, 1)

        # debug shapes
        # print("DEBUG UnitGCN shapes:", "base_A", base_A.shape, "inter_graph", inter_graph.shape, "intra_graph", intra_graph.shape, "pre_x", pre_x.shape)
        A = base_A + inter_graph + intra_graph
        out = torch.einsum("nkctv,nkcvw->nkctw", pre_x, A).contiguous()
        out = out.reshape(n, -1, t, v)
        out = self.post(out)
        return self.act(self.bn(out) + res), (inter_graph + intra_graph).reshape(n, -1, v, v)


class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        self.gcn = UnitGCN(in_channels, out_channels, A)
        self.tcn = MSTCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        res = self.residual(x)
        x, gcl_graph = self.gcn(x)
        x = self.tcn(x) + res
        return self.relu(x), gcl_graph


class ProtoGCN(nn.Module):
    def __init__(
        self,
        graph_cfg: Optional[dict] = None,
        in_channels: int = 3,
        base_channels: int = 96,
        ch_ratio: int = 2,
        num_stages: int = 10,
        inflate_stages: Sequence[int] = (5, 8),
        down_stages: Sequence[int] = (5, 8),
        data_bn_type: str = "VC",
        num_person: int = 2,
        num_classes: int = 125,
        dropout: float = 0.5,
        num_prototype: int = 100,
        **kwargs,
    ):
        super().__init__()
        graph_cfg = graph_cfg or {"dataset": "coco"}
        self.graph = Graph(**graph_cfg)
        self.data_bn_type = data_bn_type
        self.num_person = num_person
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = set(inflate_stages)
        self.down_stages = set(down_stages)
        self.num_stages = num_stages
        A = torch.tensor(self.graph.A, dtype=torch.float32)

        if data_bn_type == "MVC":
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == "VC":
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        blocks = []
        current_channels = base_channels if in_channels == base_channels else base_channels
        first_in = in_channels
        if first_in != current_channels:
            blocks.append(GCNBlock(first_in, current_channels, A.clone(), stride=1, residual=False))
        inflate_times = 0
        for stage in range(2, num_stages + 1):
            stride = 2 if stage in self.down_stages else 1
            if stage in self.inflate_stages:
                inflate_times += 1
            next_channels = int(base_channels * (ch_ratio ** inflate_times) + 1e-4)
            blocks.append(GCNBlock(current_channels, next_channels, A.clone(), stride=stride))
            current_channels = next_channels

        self.gcn = nn.ModuleList(blocks)
        self.post = nn.Conv2d(current_channels, current_channels, 1)
        self.bn = nn.BatchNorm2d(current_channels)
        self.relu = nn.ReLU(inplace=True)
        self.prn = nn.Sequential(
            nn.Linear(current_channels, num_prototype, bias=False),
            nn.Softmax(dim=-1),
            nn.Linear(num_prototype, current_channels, bias=False),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(current_channels, num_classes)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            n, m, t, v, c = x.shape
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            if self.data_bn_type == "MVC":
                x = self.data_bn(x.view(n, m * v * c, t))
            elif self.data_bn_type == "VC":
                x = self.data_bn(x.view(n * m, v * c, t))
            else:
                x = x.view(n * m, c, t, v)
                return x
            x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
            return x
        if x.ndim == 4:
            n, t, v, c = x.shape
            x = x.permute(0, 3, 1, 2).contiguous()
            return x
        raise ValueError(f"Expected 4D or 5D input, got {x.shape}")

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._normalize_input(x)
        last_graph = None
        for block in self.gcn:
            x, last_graph = block(x)
        x = self.relu(self.bn(self.post(x)))
        pooled = self.pool(x).flatten(1)
        return pooled, last_graph

    def forward(self, x: torch.Tensor, return_embedding: bool = False, return_graph: bool = False):
        feat, graph = self.forward_features(x)
        embedding = self.prn[3](self.prn[2](self.prn[1](self.prn[0](feat))))
        logits = self.fc(embedding)
        if return_graph:
            return logits, embedding, graph
        if return_embedding:
            return logits, embedding
        return logits


__all__ = ["Graph", "ProtoGCN"]