% 设置文件夹路径
folder = './data';

% 获取文件夹中所有文件的信息
file_list = dir(folder);

% 初始化文件路径变量
file_paths = {};

people_idx = 0;

% 循环遍历每个文件
for i = 1:numel(file_list)
   % 跳过当前目录和上级目录
   if strcmp(file_list(i).name,'.') || strcmp(file_list(i).name,'..')
       continue;
   end
   
   % 拼接文件路径
   file_path = fullfile(folder,file_list(i).name);
   
   % 如果是文件而不是文件夹，则将其添加到路径列表中
   if ~file_list(i).isdir
      file_paths{end+1} = file_path; 
   else % 如果是文件夹，则递归获取其下所有文件
      sub_files = get_all_files(file_path);
      sub_files = string(sub_files);
      for j=1:length(sub_files)
          dataset_string = sub_files(j);
          data_cnt = extractData(dataset_string, j, people_idx);
          fprintf("数据集："+dataset_string+" 的第"+j+"个位置共导入了"+(data_cnt+1)+"条数据");
      end
      file_paths = [file_paths, sub_files];
   end
   people_idx = people_idx + 1;
end



% 递归函数定义
function [file_paths] = get_all_files(folder)
    % 获取文件夹中所有文件的信息
    file_list = dir(folder);
    
    % 初始化文件路径变量
    file_paths = {};
    
    % 循环遍历每个文件
    for i = 1:numel(file_list)
        % 跳过当前目录和上级目录
        if strcmp(file_list(i).name,'.') || strcmp(file_list(i).name,'..')
            continue;
        end
        
        % 拼接文件路径
        file_path = fullfile(folder,file_list(i).name);
        
        % 如果是文件而不是文件夹，则将其添加到路径列表中
        if ~file_list(i).isdir
            file_paths{end+1} = file_path; 
        else % 如果是文件夹，则递归获取其下所有文件
            sub_files = get_all_files(file_path);
            file_paths = [file_paths, sub_files];
        end
    end
end