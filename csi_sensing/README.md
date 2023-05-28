

## Requirements

1. Install `pytorch` and `torchvision` (we use `pytorch==1.12.0` and `torchvision==0.13.0`).
2. `pip install -r requirements.txt`

## Run
### Download Processed Data
Please download and organize the processed datasets in this structure:
```
Benchmark
├── Data
    ├── MH_data
    │   ├── test
    │   ├── train
```

### Supervised Learning
To run models with supervised learning (train & test):  
Run: `python run.py --model [model name] --dataset [dataset name] --modal [Mag or Phase]`  

You can choose [model name] from the model list below
- MLP
- LeNet
- VGG16
- ResNet18
- ResNet50
- ResNet101
- InceptionV3
- EfficientNet
- ViT
- SwinTransformer
- mobilenetv2_035
- mobilenetv2_050
- mobilenetv2_075

You can choose [dataset name] from the dataset list below
- MH_data

*Example: `python run.py --model ResNet18 --dataset NTU-Fi_HAR --modal Mag`*
### Unsupervised Learning(Not support now)
To run models with unsupervised (self-supervised) learning (train on **NTU-Fi HAR** & test on **NTU-Fi HumanID**):  
Run: `python self_supervised.py --model [model name] ` 



## Model Zoo



## Dataset
#### Mh_data
- **CSI size** : 3 x 1992 x 40
- **number of classes** : 25
- **classes** : p0,p1,...,p24
- **train number** : 3200
- **test number** : 1175

