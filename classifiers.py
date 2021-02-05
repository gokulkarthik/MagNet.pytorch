import torch.nn as nn
import torch.nn.functional as F


class Classifier1(nn.Module):
    """Classifier used for MNIST dataset in MagNet paper
    """

    def __init__(self, in_channels=1, num_classes=10):
        super(Classifier1, self).__init__()
        self.conv_11 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_12 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_21 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_22 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dense_1 = nn.Linear(in_features=64 * 7 * 7, out_features=200)  # 3136 -> 200
        self.dense_2 = nn.Linear(in_features=200, out_features=200)
        self.out = nn.Linear(in_features=200, out_features=num_classes)

    def forward(self, X):
        """Forward Pass

        :param X: Mini-batch of shape [-1, 1, H, W]
        :return : Y_pred_logits of shape [-1, 10]
        """
        X = F.relu(self.conv_11(X))
        X = F.relu(self.conv_12(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv_21(X))
        X = F.relu(self.conv_22(X))
        X = F.max_pool2d(X, 2)
        X = X.view(X.shape[0], -1)
        X = F.relu(self.dense_1(X))
        X = F.relu(self.dense_2(X))
        X = self.out(X)

        return X


class Classifier2(nn.Module):
    """Classifier used for CIFAR dataset in MagNet paper
    """

    def __init__(self, in_channels=3, num_classes=10):
        super(Classifier2, self).__init__()
        self.conv_11 = nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.conv_12 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.conv_13 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.conv_21 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.conv_22 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.conv_23 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.conv_31 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.conv_32 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0)
        self.conv_33 = nn.Conv2d(in_channels=192, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.out = nn.Linear(in_features=10, out_features=num_classes)

    def forward(self, X):
        """Forward Pass

        :param X: Mini-batch of shape [-1, 3, H, W]
        :return : Y_pred_logits of shape [-1, 10]
        """
        X = F.relu(self.conv_11(X))
        X = F.relu(self.conv_12(X))
        X = F.relu(self.conv_13(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv_21(X))
        X = F.relu(self.conv_22(X))
        X = F.relu(self.conv_23(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv_31(X))
        X = F.relu(self.conv_32(X))
        X = F.relu(self.conv_33(X))
        X = F.avg_pool2d(X, X.shape[2])
        X = X.view(X.shape[0], -1)
        X = self.out(X)

        return X
