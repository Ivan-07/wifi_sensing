import os
import shutil

def split_files(source_dir, target_train_dir, target_test_dir):
    # 获取源文件夹中的所有文件
    files = os.listdir(source_dir)

    # 过滤以点开头的文件（隐藏文件）
    files = [file for file in files if not file.startswith('.')]

    # 计算每份文件的数量
    num_files = len(files)
    num_files_per_split = num_files // 2

    # 按每份文件数量分割文件列表
    files_split1 = files[:num_files_per_split]
    files_split2 = files[num_files_per_split:]

    # 将分割后的文件复制到目标文件夹
    for idx, file in enumerate(files_split1):
        source_file = os.path.join(source_dir, file)
        target_file = os.path.join(target_train_dir, f'new_data{idx}.mat')
        shutil.copy(source_file, target_file)

    for idx, file in enumerate(files_split2):
        source_file = os.path.join(source_dir, file)
        target_file = os.path.join(target_test_dir, f'new_data{idx}.mat')
        shutil.copy(source_file, target_file)

# 定义相关路径


# # 分割p0文件夹下的文件
# split_files(val_easy_p0_dir, mag_train_p0_dir, mag_test_p0_dir)
#
# # 分割p1文件夹下的文件
# split_files(val_easy_p1_dir, mag_train_p1_dir, mag_test_p1_dir)

for i in range(1):
    mag_train_dir = './Data/MH_data/Mag/train/p_'+str(i)+'/'
    mag_test_dir = './Data/MH_data/Mag/test/p_'+str(i)+'/'
    val_easy_dir = './Data/MH_data/val_easy/Mag/p_'+str(i)+'/'
    split_files(val_easy_dir, mag_train_dir, mag_test_dir)