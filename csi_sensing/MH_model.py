import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


class MH_MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(MH_MLP, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)  # first fully connected layer
        self.relu1 = nn.ReLU()  # ReLU activation function after fc1
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # second fully connected layer
        self.relu2 = nn.ReLU()  # ReLU activation function after fc2
        self.fc3 = nn.Linear(hidden_size, out_size)  # third fully connected layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten the input tensor
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def MLP(num_classes):
    return MH_MLP(3 * 224 * 224, 256, num_classes)


class MH_LeNet(nn.Module):
    def __init__(self, num_classes=25):
        super(MH_LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 输入通道数为3，输出通道数为6，卷积核大小为5x5
        self.pool = nn.MaxPool2d(2, 2)  # 池化核的大小为2x2，步长为2
        self.conv2 = nn.Conv2d(6, 16, 5)  # 输入通道数为6，输出通道数为16，卷积核大小为5x5
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # 两个卷积层和池化层后，输出的特征图大小为(16, 53, 53)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  # 最后一个全连接层的输出大小为10，表示需要分类10个类别

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 进行第一个卷积操作，并使用ReLU激活函数，然后进行池化操作
        x = self.pool(F.relu(self.conv2(x)))  # 进行第二个卷积操作，并使用ReLU激活函数，然后进行池化操作
        x = x.view(-1, 16 * 53 * 53)  # 将卷积层的输出展成一维向量
        x = F.relu(self.fc1(x))  # 进行第一个全连接层的计算，并使用ReLU激活函数
        x = F.relu(self.fc2(x))  # 进行第二个全连接层的计算，并使用ReLU激活函数
        x = self.fc3(x)  # 进行输出层的计算
        return x


def LeNet(num_classes):
    return MH_LeNet(num_classes)


def VGG16(num_classes):
    # 载入预训练的 VGG 模型
    model = timm.create_model('vgg16', pretrained=True, num_classes=num_classes)
    return model


def ResNet18(num_classes):
    model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)

    return model


def ResNet50(num_classes):
    model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
    return model


def ResNet101(num_classes):
    model = timm.create_model('resnet101', pretrained=True, num_classes=num_classes)
    return model


def InceptionV3(num_classes):
    # 载入预训练的 Inception 模型
    model = timm.create_model('inception_v3', pretrained=True, num_classes=num_classes)
    return model


def EfficientNet(num_classes):
    # 载入预训练的 EfficientNet 模型
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
    return model


def ViT(num_classes):
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    return model


def SwinTransformer(num_classes):
    model = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=num_classes)
    return model


def MobileNetv2_035(num_classes):
    model = timm.create_model('mobilenetv2_035', pretrained=False, num_classes=num_classes)
    return model


def MobileNetv2_050(num_classes):
    model = timm.create_model('mobilenetv2_050', pretrained=False, num_classes=num_classes)
    return model


def MobileNetv2_075(num_classes):
    model = timm.create_model('mobilenetv2_075', pretrained=False, num_classes=num_classes)
    return model


