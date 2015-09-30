% Projeto Marinha do Brasil

% Autor: Natanael Junior (natmourajr@gmail.com)
% Laboratorio de Processamento de Sinais - UFRJ


% Objetivos: Treinar (com Validacao Cruzada) SVM para classificacao de navios

fprintf('Starting %s.m\n',mfilename('fullpath'));

% path for data
workspace = getenv('MARINHA_WORKSPACE');
datapath = getenv('OUTPUTDATAPATH');
outputpath = sprintf('%s/Classification/SVM',datapath);

% Add functions Folder
addpath(genpath('../functions'));

% Load Data

% Raw Data
if(~exist(sprintf('%s/RawData.mat',datapath),'file'))
    error('DO %s/DataHandler/CreateRawData\n',workspace);
else
    load(sprintf('%s/RawData.mat',datapath));
end

% Train Information
if(~exist(sprintf('%s/TrainInformation.mat',datapath),'file'))
    error('DO %s/DataHandler/CreateRawData\n',workspace);
else
    load(sprintf('%s/TrainInformation.mat',datapath));
end

% Lofar Data
if(~exist(sprintf('%s/LofarData.mat',datapath),'file'))
    error('DO %s/DataHandler/LOFARANALYSIS\n',workspace);
else
    load(sprintf('%s/LofarData.mat',datapath));
end

data2train = [];
target2train = [];
target2train_norm = [];

for class = 1:numel(class_labels) % todas as classes
    % data
    aux = data_lofar.(class_labels{class});
    data2train   = [data2train aux];
    
    % target
    target2train = [target2train class*ones(1,length(aux))]; % numeric target
    
    aux = -1*ones(numel(class_labels),length(aux)); % all class as -1
    aux(class,:) = 1*ones(1,length(aux)); % only one class as 1
    
    target2train_norm = [target2train_norm aux];
end

CVO = cvpartition(target2train,'Kfold',n_folds);

% training parameters
normalization = 'mapstd';

save(sprintf('%s/TrainInformationSVM.mat',outputpath),'normalization','CVO');

for ifolds = 1:n_folds
    if (develop_mode) && (ifolds > 2), continue; end
    fprintf('iFold: %i of %i - Training SVM model\n',ifolds, n_folds);
    trn_id =  CVO.training(ifolds);
    tst_id =  CVO.test(ifolds);
    
    itrn = []; itst = [];
    for i = 1:length(data2train)
        if trn_id(i) == 1
            itrn = [itrn;i];
        else
            itst = [itst;i];
        end
    end
    ival = itst;
    
    % normalization
    if strcmp(normalization, 'mapstd') == 1
        % Mean = 0 and var = 1
        [~, norm_fact] = mapstd(data2train(:,itrn));
        data_norm = mapstd('apply', data2train ,norm_fact);
    else
        % Min = -1 and Max = 1
        [~, norm_fact] = mapminmax(data2train(:,itrn));
        data_norm = mapminmax('apply', data2train ,norm_fact);
    end
    
    svm_classifier = [];
    for iclass = 1:numel(class_labels)
        new_target = (target2train==iclass); new_target = (2*new_target)-1;
        svm_classifier{iclass} = svmtrain(data_norm(:,itrn)',new_target(:,itrn)','method','SMO','kktviolationlevel',0.2,'kernel_function','quadratic');
    end
    save(sprintf('%s/mat/RunSVMTrainCV_fold_%02i.mat',outputpath,ifolds),'svm_classifier');
end


