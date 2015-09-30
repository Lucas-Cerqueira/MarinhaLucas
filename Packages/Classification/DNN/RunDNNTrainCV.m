% Projeto Marinha do Brasil

% Autor: Natanael Junior (natmourajr@gmail.com)
% Laboratorio de Processamento de Sinais - UFRJ


% Objetivos: Treinar (com Validacao Cruzada) redes neurais para classificacao de navios

fprintf('Starting %s.m\n',mfilename('fullpath'));

% path for data
workspace = getenv('MARINHA_WORKSPACE');
datapath = getenv('OUTPUTDATAPATH');
outputpath = sprintf('%s/Classification/DNN',datapath);

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
    
    aux = zeros(numel(class_labels),length(aux)); % all class as -1
    aux(class,:) = 1*ones(1,length(aux)); % only one class as 1
    
    target2train_norm = [target2train_norm aux];
end

CVO = cvpartition(target2train,'Kfold',n_folds);

% training parameters
n_epochs = 200;
normalization = 'mapminmax';

save(sprintf('%s/TrainInformationDNN.mat',outputpath),'n_epochs','normalization','CVO');

for ifolds = 1:n_folds
    if (develop_mode) && (ifolds > 2), continue; end
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
    
    %Train a DNN
    sae = saesetup([size(data_norm,1) 100 100]);
    sae.ae{1}.activation_function       = 'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = 0.5;
    opts.numepochs =   50;
    opts.batchsize = length(itrn);
    sae = saetrain(sae, data_norm(:,itrn)', opts);
    
    rbm{ifolds} = sae;
    
    nn = nnsetup([size(data_norm,1) 100 100 4]);
    nn.activation_function              = 'sigm';
    nn.learningRate                     = 0.9;
    nn.W{1} = sae.ae{1}.W{1};
    
    opts.numepochs =   1000;
    opts.batchsize = length(itrn)/3;
    warning off;
    nn = nntrain(nn, data_norm(:,itrn)', target2train_norm(:,itrn)', opts);
    warning on;
    
    dnn{ifolds} = nn;
    
end

save(sprintf('%s/mat/RunDNNTrainCV.mat',outputpath),'dnn','rbm');


return;
