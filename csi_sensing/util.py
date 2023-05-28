from MH_model import *
from dataset import *
import math

from thop import profile


def load_data_n_model(dataset_name, model_name, root, modal='Phase', val='easy'):
    classes = {'UT_HAR_data': 7, 'NTU-Fi-HumanID': 14, 'NTU-Fi_HAR': 6, 'Widar': 22, 'MH_data': 25}
    input_size = {'MLP': 224, 'LeNet': 224, 'VGG16': 224, 'ResNet18': 224, 'ResNet50': 224, 'ResNet101': 224,
                  'InceptionV3': 299, 'EfficientNet': 224, 'ViT': 224, 'SwinTransformer': 384, 'MobileNetv2_035': 224,
                  'MobileNetv2_050': 224, 'MobileNetv2_075': 224}
    if dataset_name == "MH_data":
        print('using dataset: MH_data ' + modal)
        num_classes = classes['MH_data']

        train_loader = torch.utils.data.DataLoader(
            dataset=MH_CSI_Dataset(root + 'MH_data/' + modal + '/train/', modal=modal,
                                   input_size=input_size[model_name]), batch_size=4, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            dataset=MH_CSI_Dataset(root + 'MH_data/' + modal + '/test/', modal=modal,
                                   input_size=input_size[model_name]), batch_size=4, shuffle=False)
        # val_loader = torch.utils.data.DataLoader(
        #     dataset=MH_CSI_Dataset(root + 'MH_data/val_'+val+'/' + modal + '/', modal=modal), batch_size=32, shuffle=False)
        if model_name == 'MLP':
            print("using model: " + model_name)
            model = MLP(num_classes)
            train_epoch = 25
        elif model_name == 'LeNet':
            print("using model: " + model_name)
            model = LeNet(num_classes)
            train_epoch = 25
        elif model_name == 'VGG16':
            print("using model: " + model_name)
            model = VGG16(num_classes)
            train_epoch = 25
        elif model_name == 'InceptionV3':
            print("using model: " + model_name)
            model = InceptionV3(num_classes)
            train_epoch = 25
        elif model_name == 'ResNet18':
            print("using model: " + model_name)
            model = ResNet18(num_classes)
            train_epoch = 25
        elif model_name == 'EfficientNet':
            print("using model: " + model_name)
            model = EfficientNet(num_classes)
            train_epoch = 25
        elif model_name == 'ViT':
            print("using model: " + model_name)
            model = ViT(num_classes)
            train_epoch = 25
        elif model_name == 'SwinTransformer':
            print("using model: " + model_name)
            model = SwinTransformer(num_classes)
            train_epoch = 25
        elif model_name == 'MobileNetv2_035':
            print("using model: " + model_name)
            model = MobileNetv2_035(num_classes)
            train_epoch = 25
        elif model_name == 'MobileNetv2_050':
            print("using model: " + model_name)
            model = MobileNetv2_050(num_classes)
            train_epoch = 25
        elif model_name == 'MobileNetv2_075':
            print("using model: " + model_name)
            model = MobileNetv2_075(num_classes)
            train_epoch = 25

        # input = torch.randn(1, 3, input_size[model_name], input_size[model_name])
        # flops, params = profile(model, (input,))
        return train_loader, test_loader, model, train_epoch


def distances(p1_list_tensor, p2_list_tensor):
    p1_list = p1_list_tensor.cpu().numpy()
    p2_list = p2_list_tensor.cpu().numpy()
    dist_sum = 0
    for i in range(len(p1_list)):
        p1 = p1_list[i]
        p2 = p2_list[i]
        # 将点的编号转换为横向和纵向坐标
        x1 = p1 // 5 * 0.5
        y1 = p1 % 5 * 0.5
        x2 = p2 // 5 * 0.5
        y2 = p2 % 5 * 0.5

        # 计算距离差值
        dx = x2 - x1
        dy = y2 - y1

        # 计算距离
        dist = math.sqrt(dx ** 2 + dy ** 2)

        # 将距离添加到列表中
        dist_sum += dist
    return dist_sum
