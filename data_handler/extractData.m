function datasetIndex = extractData(path, sampleIdx, samplePoint, startIdx)

latest = opencsi(path);

datasetIndex = 0;

 

% %% 数据格式：JSON格式：{感兴趣的内容(固定不变的部分)+(时间戳、CSI信息)}
% 
% % 提取出固定的部分
% for i=1:length(latest)
%     data = latest(i);
%     CSIFrame = data{1}.CSI;
%     CBW = CSIFrame.CBW;
%     if CBW == 160
%         StandardHeader = data{1}.StandardHeader;
%         StandardHeader = rmfield(StandardHeader,{'Sequence','Fragment'});
%         outputData.StandardHeader = StandardHeader;
% 
%         RxSBasic = data{1}.RxSBasic;
%         RxSBasic = rmfield(RxSBasic, {'Timestamp', 'SystemTime', 'RSSI','RSSI1','RSSI2','RSSI3'});
%         outputData.RxSBasic = RxSBasic;
% 
%         CSIInfo = data{1}.CSI;
%         CSIInfo = rmfield(CSIInfo, {'CSI', 'Mag', 'Phase', 'SubcarrierIndex', 'PhaseSlope', 'PhaseIntercept'});
%         outputData.CSIInfo = CSIInfo;
%         break
%     end
% end

fprintf("目前从"+startIdx+"开始加载");


for i=1:length(latest)
    % CSI_data/phase/1/1_1
    idx = startIdx+datasetIndex;
    if sampleIdx == 3
        folder_mag =  ['../CSI_data/val/Mag/p',num2str(samplePoint-1),'/'];
        folder_phase =  ['../CSI_data/val/phase/p',num2str(samplePoint-1),'/'];
    else
        folder_mag =  ['../CSI_data/Mag/p',num2str(samplePoint-1),'/'];
        folder_phase =  ['../CSI_data/Phase/p',num2str(samplePoint-1),'/'];
    end

    if exist(folder_mag)==0
        mkdir(folder_mag); 
    end

    if exist(folder_phase)==0
        mkdir(folder_phase); 
    end

    if sampleIdx == 3
        filename_mag = ['../CSI_data/val/Mag/p',num2str(samplePoint-1),'/p',num2str(samplePoint-1),'_',num2str(idx)];
        filename_phase = ['../CSI_data/val/Phase/p',num2str(samplePoint-1),'/p',num2str(samplePoint-1),'_',num2str(idx)];
    else
        filename_mag = ['../CSI_data/Mag/p',num2str(samplePoint-1),'/p',num2str(samplePoint-1),'_',num2str(idx)];
        filename_phase = ['../CSI_data/Phase/p',num2str(samplePoint-1),'/p',num2str(samplePoint-1),'_',num2str(idx)];
    end

    if (exist(filename_mag) && exist(filename_phase))
        continue
    end
    data = latest(i);
    CSIFrame = data{1}.CSI;
    CBW = CSIFrame.CBW;
    if CBW ~= 160
        continue
    end
    datasetIndex = datasetIndex+1;

    if (idx >= 700)
        return;
    end

    if (sampleIdx == 3 && idx >= 300)
        return;
    end

    Mag = CSIFrame.Mag;
    Phase = CSIFrame.Phase;
    Mag = Mag(1:1992,:,:);
    Phase = Phase(1:1992,:,:);
    save(filename_mag, 'Mag');
    save(filename_phase, 'Phase');

%     fclose('all');

end




  





