import torch
import torch.nn as nn
import argparse
from util import load_data_n_model
import os
import time


def train(model, tensor_loader, num_epochs, learning_rate, criterion, device, args):
    print("-------------------------------")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    file_path = './output/train_' + args.model + '_' + args.modal + '.txt'
    with open(file_path, 'w') as f:
        start_time = time.time()
        acc_best = 0
        acc_best_epoch = -1
        mmse_best = float('inf')
        mmse_best_epoch = -1
        for epoch in range(num_epochs):
            folder = "./model_pth/" + args.model + "/" + args.modal
            if not os.path.exists(folder):
                os.makedirs(folder)

            file_path = "./model_pth/" + args.model + "/" + args.modal + "/" + args.dataset + "_" + args.model + "_model_epoch" + str(
                epoch) + ".pth"
            # if os.path.exists(file_path):
            #     model.load_state_dict(torch.load(file_path))
            #     continue
            epoch_start_time = time.time()
            model.train()
            epoch_loss = 0
            epoch_accuracy = 0
            mmse = 0
            for data in tensor_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.type(torch.LongTensor)

                optimizer.zero_grad()
                outputs = model(inputs)
                outputs = outputs.to(device)
                outputs = outputs.type(torch.FloatTensor)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                predict_y = torch.argmax(outputs, dim=1).to(device)
                epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
                mmse += torch.abs(predict_y - labels.to(device)).sum().item() / labels.size(0) * 0.25
            epoch_end_time = time.time()
            cost_time = epoch_end_time - epoch_start_time
            mmse = mmse / len(tensor_loader)
            epoch_loss = epoch_loss / len(tensor_loader.dataset)
            epoch_accuracy = epoch_accuracy / len(tensor_loader)
            if epoch_accuracy > acc_best:
                acc_best = epoch_accuracy
                acc_best_epoch = epoch
            if mmse < mmse_best:
                mmse_best = mmse
                mmse_best_epoch = epoch
            res = 'Epoch:{}, Accuracy:{:.4f},Loss:{:.9f},Cost_time:{:.4f},Mmse:{:.4f}'.format(epoch, float(epoch_accuracy),
                                                                                      float(epoch_loss),
                                                                                      float(cost_time), float(mmse))
            print(res)
            f.write(res + '\n')
            if (epoch + 1) % 1 == 0:
                torch.save(model.state_dict(), file_path)
        end_time = time.time()
        final_print = 'acc_best:{}, acc_best_epoch:{}, mmse_best:{}, mmse_best_epoch:{}, cost_time_average:{:.4f}'.format(
            float(acc_best),
            float(acc_best_epoch),
            float(mmse_best),
            float(mmse_best_epoch),
            float((end_time - start_time) / num_epochs))
        print(final_print)
        f.write(final_print + '\n')
    return


def my_test(model, tensor_loader, criterion, device, args):
    print("-------------------------------")
    folder_path = "./model_pth/" + args.model + "/" + args.modal + "/"
    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        files = sorted(files, key=lambda x: int(x.split('_epoch')[-1].split('.')[0]))
        file_path = './output/test_' + args.model + '_' + args.modal + '.txt'
        with open(file_path, 'w') as f:
            start_time = time.time()
            acc_best = 0
            acc_best_epoch = -1
            mmse_best = float('inf')
            mmse_best_epoch = -1
            epoch = 0
            for filename in files:
                model_path = os.path.join(root, filename)

                # 加载模型
                epoch_start_time = time.time()
                model.load_state_dict(torch.load(model_path))

                model.eval()
                test_acc = 0
                test_loss = 0
                mmse = 0
                with torch.no_grad():
                    for data in tensor_loader:
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels.to(device)
                        labels = labels.type(torch.LongTensor)

                        outputs = model(inputs)
                        outputs = outputs.type(torch.FloatTensor)
                        outputs.to(device)

                        loss = criterion(outputs, labels)
                        predict_y = torch.argmax(outputs, dim=1).to(device)
                        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
                        test_acc += accuracy
                        test_loss += loss.item() * inputs.size(0)
                epoch_end_time = time.time()
                cost_time = epoch_end_time - epoch_start_time
                mmse = mmse / len(tensor_loader)
                test_acc = test_acc / len(tensor_loader)
                test_loss = test_loss / len(tensor_loader.dataset)
                if test_acc > acc_best:
                    acc_best = test_acc
                    acc_best_epoch = epoch
                if mmse < mmse_best:
                    mmse_best = mmse
                    mmse_best_epoch = epoch
                res = args.model + " " + args.modal + " " + model_path.split('_')[-1][
                                                            :-4] + " test accuracy:{:.4f}, loss:{:.5f}, cost_time:{:.4f},mmse:{:.4f}".format(
                    float(test_acc), float(test_loss), float(cost_time), float(mmse))
                print(res)
                f.write(res + '\n')
                epoch += 1
            end_time = time.time()
            final_print = 'acc_best:{}, acc_best_epoch:{}, mmse_best:{}, mmse_best_epoch:{}, cost_time_average:{:.4f}'.format(
                float(acc_best),
                float(acc_best_epoch),
                float(mmse_best),
                float(mmse_best_epoch),
                float((end_time - start_time) / (epoch+1)))
            print(final_print)
            f.write(final_print + '\n')
    return


def my_val(model, tensor_loader, criterion, device, args):
    print("-------------------------------")
    print("val: " + args.val)
    folder_path = "./model_pth/" + args.model + "/" + args.modal + "/"
    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        files = sorted(files, key=lambda x: int(x.split('_epoch')[-1].split('.')[0]))
        file_path = './output/val_' + args.val + '_' + args.model + '_' + args.modal + '.txt'
        with open(file_path, 'w') as f:
            for filename in files:
                model_path = os.path.join(root, filename)

                # 加载模型
                model.load_state_dict(torch.load(model_path))

                model.eval()
                test_acc = 0
                test_loss = 0
                for data in tensor_loader:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels.to(device)
                    labels = labels.type(torch.LongTensor)

                    outputs = model(inputs)
                    outputs = outputs.type(torch.FloatTensor)
                    outputs.to(device)

                    loss = criterion(outputs, labels)
                    predict_y = torch.argmax(outputs, dim=1).to(device)
                    accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
                    test_acc += accuracy
                    test_loss += loss.item() * inputs.size(0)
                test_acc = test_acc / len(tensor_loader)
                test_loss = test_loss / len(tensor_loader.dataset)
                res = args.model + " " + args.modal + " " + model_path.split('_')[-1][
                                                            :-4] + " val accuracy:{:.4f}, loss:{:.5f}".format(
                    float(test_acc), float(test_loss))
                print(res)
                f.write(res + '\n')
    return


def test(model, tensor_loader, criterion, device, args):
    model.eval()
    test_acc = 0
    test_loss = 0
    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)

        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)

        loss = criterion(outputs, labels)
        predict_y = torch.argmax(outputs, dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)
    test_acc = test_acc / len(tensor_loader)
    test_loss = test_loss / len(tensor_loader.dataset)
    print(
        args.model + " " + args.modal + " test accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc), float(test_loss)))
    return


def main():
    root = './Data/'
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices=['UT_HAR_data', 'NTU-Fi-HumanID', 'NTU-Fi_HAR', 'Widar', 'MH_data'])
    parser.add_argument('--model',
                        choices=['MLP', 'LeNet', 'ResNet18', 'ResNet50', 'ResNet101', 'RNN', 'GRU', 'LSTM', 'BiLSTM',
                                 'CNN+GRU', 'ViT', 'EfficientNet'])
    parser.add_argument('--modal', choices=['Mag', 'Phase'])
    parser.add_argument('--val', choices=['easy', 'medium', 'hard'])
    args = parser.parse_args()

    train_loader, test_loader, val_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root,
                                                                                  args.modal, args.val)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(
        model=model,
        tensor_loader=train_loader,
        num_epochs=train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device,
        args=args
    )
    # test(
    #     model=model,
    #     tensor_loader=test_loader,
    #     criterion=criterion,
    #     device=device,
    #     args=args
    # )
    my_test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device,
        args=args
    )
    # my_val(
    #     model=model,
    #     tensor_loader=val_loader,
    #     criterion=criterion,
    #     device=device,
    #     args=args
    # )
    return


if __name__ == "__main__":
    main()
