"""Public minimal model definition for HypnoConv-BF-v1."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride=1, dilation=1, groups=1, act=True):
        super().__init__()
        padding = (kernel - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_c, out_c, kernel, stride, padding, dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm1d(out_c)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MultiBandStemBF(nn.Module):
    def __init__(self, in_channels=2, out_per_band=12, stride=6):
        super().__init__()
        bands = [15, 31, 63, 127]
        self.branches = nn.ModuleList([
            ConvBNAct(in_channels, out_per_band, k, stride=stride)
            for k in bands
        ])
        total = out_per_band * len(bands)
        self.post = nn.Sequential(
            nn.BatchNorm1d(total),
            nn.GELU(),
            nn.Conv1d(total, total, 1, bias=False),
            nn.BatchNorm1d(total),
            nn.GELU(),
        )
        self.pool = nn.MaxPool1d(8, 8)
        self.out_channels = total

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        out = torch.cat(feats, dim=1)
        out = self.post(out)
        out = self.pool(out)
        return out


class DepthwiseResBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=7, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBNAct(in_c, in_c, kernel, stride, groups=in_c)
        self.point1 = nn.Conv1d(in_c, out_c, 1, bias=False)
        self.conv2_act = ConvBNAct(out_c, out_c, kernel, groups=out_c)
        self.point2 = nn.Conv1d(out_c, out_c, 1, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.downsample = nn.Conv1d(in_c, out_c, 1, stride, bias=False) if (in_c != out_c or stride != 1) else nn.Identity()
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.point1(out)
        out = F.gelu(out)
        out = self.drop(out)
        out = self.conv2_act(out)
        out = self.point2(out)
        out = self.bn(out)
        return F.gelu(out + residual)


class TCNBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, dilation, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_c, out_c, kernel, dilation=dilation),
            nn.Dropout(dropout),
            ConvBNAct(out_c, out_c, kernel, dilation=dilation, act=False),
        )
        self.downsample = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return F.gelu(self.drop(self.net(x)) + self.downsample(x))


class TCNHead(nn.Module):
    def __init__(self, in_dim, hidden, out_dim=5, kernel=5, dilations=None, dropout=0.35):
        super().__init__()
        dilations = dilations or [1, 2, 4, 8, 16]
        self.proj = nn.Conv1d(in_dim, hidden, 1)
        self.tcn = nn.ModuleList([
            TCNBlock(hidden, hidden, kernel, d, dropout)
            for d in dilations
        ])
        self.out = nn.Conv1d(hidden, out_dim, 1)

    def forward(self, x):
        x = self.proj(x)
        for block in self.tcn:
            x = block(x)
        return self.out(x)


class HypnoConvBF(nn.Module):
    def __init__(self, in_channels=2, num_classes=5):
        super().__init__()
        self.stem = MultiBandStemBF(in_channels=in_channels, out_per_band=12, stride=6)
        stem_out = self.stem.out_channels
        self.encoder = nn.ModuleList([
            DepthwiseResBlock(stem_out, 64, kernel=7, dropout=0.4),
            DepthwiseResBlock(64, 64, kernel=7),
            nn.MaxPool1d(4, 4),
            nn.Dropout(0.4),
            DepthwiseResBlock(64, 96, kernel=5),
            DepthwiseResBlock(96, 96, kernel=5),
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = TCNHead(96, 96, out_dim=num_classes, dilations=[1, 2, 4, 8, 16])

    def forward(self, x):
        bsz, seq_len, channels, samples = x.shape
        x = x.reshape(bsz * seq_len, channels, samples)
        x = self.stem(x)
        for block in self.encoder:
            x = block(x)
        x = self.pool(x)
        x = x.reshape(bsz, seq_len, -1).transpose(1, 2)
        x = self.head(x)
        return x.transpose(1, 2)


def create_model(in_channels=2, num_classes=5):
    return HypnoConvBF(in_channels=in_channels, num_classes=num_classes)


CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
NUM_CLASSES = 5
