import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import resnet50


class ResNet18_size2(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18_size2, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x = x.reshape(-1, 4, 1992, 10)
        # x = self.resize_input(x)
        x = self.resnet(x)
        return x

    def resize_input(self, input_data):
        # 调整输入数据的大小
        resized_data = []
        for i in range(len(input_data)):
            # 转置输入数据以匹配ResNet18模型的输入
            img = input_data[i].permute(1, 2, 0)
            img = F.interpolate(img, size=(224, 224))
            # 将所有通道的均值标准化为0.5，标准差为0.5
            mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            img = (img - mean) / std
            # 将数据转换为tensor
            img = img.permute(2, 0, 1)
            resized_data.append(img)

        # 将调整大小后的数据串联为一个张量
        resized_data = torch.stack(resized_data)

        return resized_data


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out


class MH_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(MH_ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def MH_ResNet18(num_classes):
    return MH_ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    # return ResNet18(num_classes, 1)
    # return ResNet18(num_classes)


def MH_ResNet50(num_classes):
    return ResNet50(num_classes=num_classes)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=35):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * 4
        for i in range(num_blocks - 1):
            layers.append(ResidualBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        self.in_channels = 64
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.backbone = EfficientNet.from_name('efficientnet-b0')
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.backbone.extract_features(x)
        x = self.backbone._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


def MH_EfficientNet(num_classes):
    # 创建EfficientNet模型
    model = EfficientNet.from_pretrained('efficientnet-b0')

    # 设置模型输出的分类数
    num_classes = 25
    model._fc = nn.Linear(1280, num_classes)
    return model

class ResNet50_size2(nn.Module):
    def __init__(self, num_classes=35):
        super(ResNet50_size2, self).__init__()
        self.resnet = models.resnet50(pretrained=False) # 使用预训练的ResNet50
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 使用自适应平均池化代替全局平均池化
        self.fc = nn.Linear(2048, num_classes) # 输出层

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x