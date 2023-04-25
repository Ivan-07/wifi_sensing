clc;clear;
% 
% % 设置文件夹路径
% folder = './data';
% 
% % 获取文件夹中所有文件的信息
% file_list = dir(folder);
% 
% people_idx = 1;
% % 循环遍历每个文件
% for i = 7:7
%    % 跳过当前目录和上级目录
%    if strcmp(file_list(i).name,'.') || strcmp(file_list(i).name,'..')
%        continue;
%    end
%    
%    % 拼接子文件夹路径
%    folder_path = fullfile(folder,file_list(i).name);
% 
%    sub_files = dir(folder_path);
% 
%    position_idx = 0;
% %    for j=1:length(sub_files)
%    for j = 17:27
%       if strcmp(sub_files(j).name,'.') || strcmp(sub_files(j).name,'..')
%         continue;
%       end
%       dataset_path = fullfile(folder_path, sub_files(j).name);
%       data_cnt = extractData(dataset_path, j-3, i-3);
%       fprintf("数据集："+file_list(i).name+" 的第"+(j-3)+"个位置共导入了"+(data_cnt)+"条数据");
%       position_idx = position_idx+1;
%   end
%    people_idx = people_idx + 1;
% end
% 
% 
% 
% % 递归函数定义
% function [file_paths] = get_all_files(folder)
%     % 获取文件夹中所有文件的信息
%     file_list = dir(folder);
%     
%     % 初始化文件路径变量
%     file_paths = {};
%     
%     % 循环遍历每个文件
%     for i = 1:numel(file_list)
%         % 跳过当前目录和上级目录
%         if strcmp(file_list(i).name,'.') || strcmp(file_list(i).name,'..')
%             continue;
%         end
%         
%         % 拼接文件路径
%         file_path = fullfile(folder,file_list(i).name);
%         
%         % 如果是文件而不是文件夹，则将其添加到路径列表中
%         if ~file_list(i).isdir
%             file_paths{end+1} = file_path; 
%         else % 如果是文件夹，则递归获取其下所有文件
%             sub_files = get_all_files(file_path);
%             file_paths = [file_paths, sub_files];
%         end
%     end
% end


folder = './val_easy';
sub_files = dir(folder);

for j=1:length(sub_files)
%    for j = 25:27
  if strcmp(sub_files(j).name,'.') || strcmp(sub_files(j).name,'..')
    continue;
  end
  dataset_path = fullfile(folder, sub_files(j).name);
  data_cnt = extractData(dataset_path, j-3, 0);
  fprintf("数据集：czr"+" 的第"+(j-3)+"个位置共导入了"+(data_cnt)+"条数据");
end