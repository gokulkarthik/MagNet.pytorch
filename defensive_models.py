import torch
import torch.nn as nn
import torch.nn.functional as F


class DefensiveModel1(nn.Module):
    """Defensive model used for MNIST in MagNet paper
    """

    def __init__(self, in_channels=1):
        super(DefensiveModel1, self).__init__()
        self.conv_11 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv_21 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv_22 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv_31 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv_32 = nn.Conv2d(in_channels=3, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        """Forward propagation

        :param X: Mini-batch of shape [-1, 1, H, W]
        :return: Mini-batch of shape [-1, 1, H, W]
        """
        X = torch.sigmoid(self.conv_11(X))
        X = F.avg_pool2d(X, 2)
        X = torch.sigmoid(self.conv_21(X))
        X = torch.sigmoid(self.conv_22(X))
        X = F.interpolate(X, scale_factor=2)
        X = torch.sigmoid(self.conv_31(X))
        X = torch.sigmoid(self.conv_32(X))

        return X


class DefensiveModel2(nn.Module):
    """Defensive model used for CIFAR-10 in MagNet paper
    """

    def __init__(self, in_channels=3):
        super(DefensiveModel2, self).__init__()
        self.conv_11 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv_21 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv_31 = nn.Conv2d(in_channels=3, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        """Forward propagation

        :param X: Mini-batch of shape [-1, 1, H, W]
        :return: Mini-batch of shape [-1, 1, H, W]
        """
        X = torch.sigmoid(self.conv_11(X))
        X = torch.sigmoid(self.conv_21(X))
        X = torch.sigmoid(self.conv_31(X))

        return X