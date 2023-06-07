import shutil

models = ["MLP", "LeNet", "VGG16", "ResNet18", "ResNet50", "ResNet101", "InceptionV3", "EfficientNet", "ViT",
               "MobileNetv2_035", "MobileNetv2_050", "MobileNetv2_075"]


for model in models:
    txt_path = 'output/test_'+model+'_Mag.txt'
    with open(txt_path, 'r') as file:
        content = file.read()

    content_list = content.split(',')
    for item in content_list:
        if "acc_best_epoch" in item:
            acc_best_epoch = item.strip().split(':')[1]

    source_path = 'model_pth/'+model+'/Mag/'+'MH_data_'+model+'_model_epoch'+acc_best_epoch+'.pth'
    aim_dir = 'model/'+model+'/Mag/'

    shutil.move(source_path, aim_dir)

