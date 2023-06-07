import time

import torch
from dataset import MH_CSI_Dataset
from MH_model import *
from util import distances
from thop import profile

models = ["MLP", "LeNet", "VGG16", "ResNet18", "ResNet50", "ResNet101", "InceptionV3", "EfficientNet", "ViT",
          "MobileNetv2_035", "MobileNetv2_050", "MobileNetv2_075"]

input_size = {'MLP': 224, 'LeNet': 224, 'VGG16': 224, 'ResNet18': 224, 'ResNet50': 224, 'ResNet101': 224,
              'InceptionV3': 299, 'EfficientNet': 224, 'ViT': 224, 'SwinTransformer': 384, 'MobileNetv2_035': 224,
              'MobileNetv2_050': 224, 'MobileNetv2_075': 224}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


# 指标：准确度、耗时、mmse、params、flaps

def test(model, model_name):
    test_loader = torch.utils.data.DataLoader(
        dataset=MH_CSI_Dataset('./Data/MH_data/Mag/test/', modal='Mag',
                               input_size=input_size[model_name]), batch_size=64, shuffle=False)
    model_path = 'models/' + model_name + '.pth'

    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_acc = 0
    mmse = 0
    time_cost = 0
    with torch.no_grad():
        for data in test_loader:
            start_time = time.time()
            inputs, labels = data
            inputs = inputs.to(device)
            labels.to(device)
            labels = labels.type(torch.LongTensor)

            outputs = model(inputs)
            outputs = outputs.type(torch.FloatTensor)
            outputs.to(device)

            predict_y = torch.argmax(outputs, dim=1).to(device)
            end_time = time.time()

            accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
            test_acc += accuracy
            mmse += distances(predict_y, labels.to(device)) / labels.size(0)
            time_cost += (end_time - start_time) / labels.size(0)
    mmse = mmse / len(test_loader)
    test_acc = test_acc / len(test_loader)
    time_cost = time_cost / len(test_loader)
    return mmse, test_acc, time_cost

def get_flaps_params(model, model_name):
    flops, params = profile(model, input_size=(1, 3, input_size[model_name], input_size[model_name]))

    return flops, params

if __name__ == '__main__':
    model_name = 'MLP'
    model = eval(model_name)(25)
    mmse, test_acc, time_cost = test(model, model_name)
    print('Acc = {:.4f}' % float(test_acc))
    print('Mmse = {:.4f}' % float(mmse))
    print('Time = {:.4f}' % float(time_cost))
    flops, params = get_flaps_params(model, model_name)
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')