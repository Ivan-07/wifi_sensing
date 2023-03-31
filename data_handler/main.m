clc;clear;

% 加载工具包的位置
addpath('D:\softwares\matlab2021\toolbox\jsonlab-master');

startIdx = zeros(35, 1);
% for i=2:3
for i=1:2
    path = ['..\sample_data\sample_data', num2str(i),'\'];                   % 设置数据存放的文件夹路径
    file = dir(fullfile(path,'*.csi'));  % 显示文件夹下所有符合后缀名为.txt文件的完整信息
    fileNames = {file.name}';            % 提取符合后缀名为.txt的所有文件的文件名，转换为n行1列

    lengthNames = size(fileNames,1);    % 获取所提取数据文件的个数
    for k = 1 : lengthNames
        % 连接路径和文件名得到完整的文件路径
        kPath = strcat(path, file(k).name);
        kNum = extractData(kPath, i, k, startIdx(k));
        fprintf("第"+i+"批数据的第"+k+"个位置共导入了"+kNum+"条数据");
        fprintf('\n');
        startIdx(k) = startIdx(k) + kNum;
    end
    fclose('all');
end