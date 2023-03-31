clc;clear;

% filename_pre = 'CSI_data/phase/p0/p0_';

for i=1:2
    for j=1:5
        rng(0);
        disp(randperm(5))
    end
end

% for i=0:19
%     filename  = [filename_pre, num2str(i)];
%     load(filename)
% %     x = 1:size(Phase, 1);
%     x = 1:1992;
%     plot(x, Phase(1:1992,1,1));
%     hold on;
% end