"""
model.py — PyTorch 精确复刻 DSP 端 BC-ResNet 架构

架构来源: Code/project3.h → Project3_BCResNetForward()
输入: Log-Mel spectrogram [batch, 1, 40, 101]
输出: 12 类 logits
"""

import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    """
    与 Project3_BCResBlock() 逐位对应:
      expand: 1x1 Conv → BN → ReLU
      depthwise: 3x3 DWConv → BN → ReLU
      project: 1x1 Conv → BN (无激活)
      shortcut: 1x1 Conv → BN (无激活), 当 in_c≠out_c 或 stride≠1 时使用
      add → ReLU
    """
    def __init__(self, in_c: int, out_c: int, stride: int = 1, bn_eps: float = 1e-5):
        super().__init__()
        self.use_shortcut = (in_c != out_c) or (stride != 1)

        # expand: in_c → out_c, 1×1
        self.expand = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c, eps=bn_eps)

        # depthwise: out_c → out_c, 3×3
        self.dw = nn.Conv2d(out_c, out_c, kernel_size=3, stride=stride,
                            padding=1, groups=out_c, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c, eps=bn_eps)

        # project: out_c → out_c, 1×1 (no activation)
        self.project = nn.Conv2d(out_c, out_c, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c, eps=bn_eps)

        # shortcut: in_c → out_c, 1×1 (no activation)
        if self.use_shortcut:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1,
                                      stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_c, eps=bn_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.expand(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.dw(out)
        out = self.bn2(out)
        out = torch.relu(out)

        out = self.project(out)
        out = self.bn3(out)

        if self.use_shortcut:
            identity = self.shortcut(identity)
            identity = self.shortcut_bn(identity)

        out = out + identity
        out = torch.relu(out)
        return out


class BcResNet(nn.Module):
    """
    精确复刻 DSP 端 Project3_BCResNetForward() 的完整前向流程.
    """
    def __init__(self, n_classes: int = 12, bn_eps: float = 1e-5):
        super().__init__()

        # conv1: 1→16, 3×3, stride=(2,1), pad=(1,1)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=(2, 1),
                               padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16, eps=bn_eps)

        # BC-ResBlock × 3
        self.block1 = InvertedResidual(16, 8, stride=1, bn_eps=bn_eps)
        self.block2 = InvertedResidual(8, 12, stride=(2, 1), bn_eps=bn_eps)
        self.block3 = InvertedResidual(12, 16, stride=(2, 1), bn_eps=bn_eps)

        # dwconv: 16→16, 3×3, same size
        self.dwconv = nn.Conv2d(16, 16, kernel_size=3, stride=1,
                                padding=1, groups=16, bias=False)
        self.bn_dw = nn.BatchNorm2d(16, eps=bn_eps)

        # pwconv: 16→20, 1×1
        self.pwconv = nn.Conv2d(16, 20, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(20, eps=bn_eps)

        # conv2: 20→20, kernel=(5,1), stride=1
        self.conv2 = nn.Conv2d(20, 20, kernel_size=(5, 1), stride=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(20, eps=bn_eps)

        # expand: 20→32, 1×1
        self.expand = nn.Conv2d(20, 32, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(32, eps=bn_eps)

        # classifier: 32→n_classes
        self.fc = nn.Linear(32, n_classes, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming 初始化 (fine-tune 前使用，加载预训练权重后会覆盖)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, 40, 101] log-mel spectrogram
        Returns:
            logits: [B, n_classes]
        """
        # conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)           # → [B, 16, 20, 101]

        # BC-ResBlocks
        x = self.block1(x)          # → [B, 8,  20, 101]
        x = self.block2(x)          # → [B, 12, 10, 101]
        x = self.block3(x)          # → [B, 16, 5,  101]

        # dwconv + BN + ReLU
        x = self.dwconv(x)
        x = self.bn_dw(x)
        x = torch.relu(x)           # → [B, 16, 5, 101]

        # pwconv + BN + ReLU
        x = self.pwconv(x)
        x = self.bn_pw(x)
        x = torch.relu(x)           # → [B, 20, 5, 101]

        # conv2 + BN + ReLU
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)           # → [B, 20, 1, 101]

        # expand + BN + ReLU
        x = self.expand(x)
        x = self.bn_expand(x)
        x = torch.relu(x)           # → [B, 32, 1, 101]

        # global average pooling over time dimension
        x = x.mean(dim=-1)          # → [B, 32, 1, 1]
        x = x.view(x.size(0), -1)   # → [B, 32]

        # classifier
        x = self.fc(x)              # → [B, n_classes]
        return x


def count_parameters(model: nn.Module) -> dict:
    """统计模型参数量."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    model = BcResNet(n_classes=13)
    print(f"Parameters: {count_parameters(model)}")
    x = torch.randn(1, 1, 40, 101)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape} (expected: [1, 13])")

    # 逐层打印 shape
    print("\n--- Layer shapes ---")
    x = torch.relu(model.bn1(model.conv1(x)))
    print(f"conv1+BN+ReLU:    {list(x.shape)}")

    x = model.block1(x)
    print(f"block1:           {list(x.shape)}")

    x = model.block2(x)
    print(f"block2:           {list(x.shape)}")

    x = model.block3(x)
    print(f"block3:           {list(x.shape)}")

    x = torch.relu(model.bn_dw(model.dwconv(x)))
    print(f"dwconv+BN+ReLU:   {list(x.shape)}")

    x = torch.relu(model.bn_pw(model.pwconv(x)))
    print(f"pwconv+BN+ReLU:   {list(x.shape)}")

    x = torch.relu(model.bn2(model.conv2(x)))
    print(f"conv2+BN+ReLU:    {list(x.shape)}")

    x = torch.relu(model.bn_expand(model.expand(x)))
    print(f"expand+BN+ReLU:   {list(x.shape)}")

    x = x.mean(dim=-1)
    print(f"avgpool (time):   {list(x.shape)}")

    x = x.view(x.size(0), -1)
    y = model.fc(x)
    print(f"fc:               {list(y.shape)}")
