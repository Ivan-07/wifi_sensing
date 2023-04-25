clc;clear all
strings = ["Mag", "Phase"];
dataset_len = 580;
for k=1:2
    string = strings(k);
    for idx=0:24
        % 指定原始文件夹的路径
        folder_path = '../CSI_data/'+string+'/p_'+num2str(idx);
        
        % 获取所有.mat文件的列表
        mat_files = dir(fullfile(folder_path, '*.mat'));
        
        % 设置随机数种子，以便每次运行结果相同
        rng(0);
        
        % 生成一个随机的索引序列
%         rand_idx = randperm(length(mat_files));
        rand_idx = randperm(dataset_len);
        % 将索引序列拆分成两个部分
        split_ratio = 0.9;
        split_idx = round(split_ratio * dataset_len);
        split1 = rand_idx(1:split_idx);
        split2 = rand_idx(split_idx+1:end);
        
        % 创建两个新的文件夹来保存.mat文件
        folder1_path = '../MH_data/'+string+'/train/p_'+num2str(idx);
        folder2_path = '../MH_data/'+string+'/test/p_'+ num2str(idx);
        if exist(folder1_path)==0
            mkdir(folder1_path); 
        end
        if exist(folder2_path)==0
            mkdir(folder2_path); 
        end
        
        % 保存第一部分的.mat文件到新文件夹1中
        for i = 1:length(split1)
            filename = mat_files(split1(i)).name;
            source_path = fullfile(folder_path, filename);
            dest_path = fullfile(folder1_path, filename);
            copyfile(source_path, dest_path);
        end
        
        % 保存第二部分的.mat文件到新文件夹2中
        for i = 1:length(split2)
            filename = mat_files(split2(i)).name;
            source_path = fullfile(folder_path, filename);
            dest_path = fullfile(folder2_path, filename);
            copyfile(source_path, dest_path);
        end
    end
end