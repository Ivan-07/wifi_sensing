import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def MH_MLP(num_classes):
    return MLP(num_classes)


def MH_LeNet(num_classes):
    return LeNet(num_classes)


def MH_ResNet18(num_classes):
    return ResNet18(num_classes)


def MH_ResNet50(num_classes):
    return ResNet50(num_classes=num_classes)


def MH_ResNet101(num_classes):
    return ResNet101(num_classes=num_classes)


def MH_RNN(num_classes):
    return RNN(num_classes)


def MH_GRN(num_classes):
    return GRU(num_classes)


def MH_LSTM(num_classes):
    return LSTM(num_classes)


def MH_BiLSTM(num_classes):
    return BiLSTM(num_classes)


class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4 * 166 * 120, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(-1, 4 * 166 * 120)
        x = self.fc(x)
        x = self.classifier(x)
        return x


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, (3, 5), stride=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, (3, 3), stride=(1, 3)),
            nn.ReLU(True),
            nn.Conv2d(64, 96, (3, 3), stride=(1, 3)),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(184320, 128),  # Adjusted input size
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.resnet(x)


class ResNet101(nn.Module):
    def __init__(self, num_classes):
        super(ResNet101, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.resnet(x)


class RNN(nn.Module):
    def __init__(self, num_classes):
        super(RNN, self).__init__()
        # input size (4, 166, 120)
        self.rnn = nn.RNN(166, 64, num_layers=1)  # Update input size to 166
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, 166, 120)  # Update view dimensions to (166, 120)
        x = x.permute(2, 0, 1)  # Update permute dimensions
        _, ht = self.rnn(x)
        outputs = self.fc(ht[-1])
        return outputs


class GRU(nn.Module):
    def __init__(self, num_classes):
        super(GRU, self).__init__()
        self.gru = nn.GRU(166, 64, num_layers=1, batch_first=True)  # Update the input size and set batch_first=True
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute dimensions to (batch_size, 120, 166)
        _, ht = self.gru(x)
        outputs = self.fc(ht.squeeze(0))  # Squeeze the batch dimension
        return outputs


class LSTM(nn.Module):
    def __init__(self, num_classes):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(166, 64, num_layers=1, batch_first=True)  # Update the input size and set batch_first=True
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute dimensions to (batch_size, 120, 166)
        _, (ht, ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs


class BiLSTM(nn.Module):
    def __init__(self, num_classes):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(166, 64, num_layers=1, bidirectional=True,
                            batch_first=True)  # Update the input size and set batch_first=True
        self.fc = nn.Linear(128, num_classes)  # Double the input size for bidirectional LSTM

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute dimensions to (batch_size, 120, 166)
        _, (ht, ct) = self.lstm(x)
        outputs = self.fc(torch.cat((ht[-2], ht[-1]), dim=1))  # Concatenate hidden states from both directions
        return outputs
