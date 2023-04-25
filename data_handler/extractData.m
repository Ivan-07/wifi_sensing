function idx = extractData(path, samplePoint, people_idx)

latest = opencsi(path);

idx = 0;
batch_size = 10;
 
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


batch_mag = [];
batch_phase = [];
for i=1:length(latest)

    folder_mag =  ['../CSI_data/val_easy/Mag/p_',num2str(samplePoint),'/'];
    folder_phase =  ['../CSI_data/val_easy/Phase/p_',num2str(samplePoint),'/'];

    if exist(folder_mag)==0
        mkdir(folder_mag); 
    end

    if exist(folder_phase)==0
        mkdir(folder_phase); 
    end

    filename_mag = ['../CSI_data/val_easy/Mag/p_',num2str(samplePoint),'/p_',num2str(samplePoint),'_',num2str(people_idx),'_',num2str(floor(idx/batch_size))];
    filename_phase = ['../CSI_data/val_easy/Phase/p_',num2str(samplePoint),'/p_',num2str(samplePoint),'_',num2str(people_idx),'_',num2str(floor(idx/batch_size))];

%     if (exist(filename_mag) && exist(filename_phase))
%         continue
%     end
    data = latest(i);
    CSIFrame = data{1}.CSI;
    CBW = CSIFrame.CBW;
    if CBW ~= 160
        continue
    end
    idx = idx+1;


    Mag = CSIFrame.Mag;
    Phase = CSIFrame.Phase;
    Mag = Mag(1:1992,:,:);
    Phase = Phase(1:1992,:,:);
    Mag = reshape(Mag, [1992, 1, 4]);
    Phase = reshape(Phase, [1992, 1, 4]);

    batch_mag = [batch_mag, Mag];
    batch_phase = [batch_phase, Phase];

    if mod(idx, batch_size) == 0
        Mag = batch_mag;
        Phase = batch_phase;
        save(filename_mag, 'Mag');
        save(filename_phase, 'Phase');
        batch_mag = [];
        batch_phase = [];
    end

%     fclose('all');

end

idx = floor(idx/batch_size);



  





