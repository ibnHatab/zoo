
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, num_channels, use_1x1conv=False, stride=1) -> None:
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

class ResNet18(nn.Module):
    def __init__(self, num_classes, in_channels=1) -> None:
        super().__init__()

        def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
            blk = []
            for i in range(num_residuals):
                if i == 0 and not first_block:
                    blk.append(Residual(out_channels, use_1x1conv=True, stride=2))
                else:
                    blk.append(Residual(out_channels))

            return nn.Sequential(*blk)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),)
        self.net.add_module('resnet_block1', resnet_block(64, 64, 2, first_block=True))
        self.net.add_module('resnet_block2', resnet_block(64, 128, 2))
        self.net.add_module('resnet_block3', resnet_block(128, 256, 2))
        self.net.add_module('resnet_block4', resnet_block(256, 512, 2))
        self.net.add_module('global_avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
        self.net.add_module('fc', nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))

    def forward(self, X):
        return self.net(X)

