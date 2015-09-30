
% Projeto Marinha do Brasil

% Autor: Natanael Junior (natmourajr@gmail.com)
% Laboratorio de Processamento de Sinais - UFRJ


% Objetivos: Treinar (com Validacao Cruzada) redes neurais para classificacao de navios

fprintf('Starting %s.m\n',mfilename('fullpath'));

% path for data
workspace = getenv('MARINHA_WORKSPACE'); 
datapath = getenv('OUTPUTDATAPATH');
outputpath = sprintf('%s/Classification/NeuralNetwork',datapath);

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

possible_topo = 2:3;

% training parameters
train_fnc = 'trainlm'; % weights update function
perf_fnc = 'mse'; % error function
act_fnc = {'tansig' 'tansig'}; % activation function
n_epochs = 200;
show = true;
normalization = 'mapstd';

save(sprintf('%s/TrainInformationNN.mat',outputpath),'possible_topo','train_fnc','perf_fnc','act_fnc','n_epochs','normalization','CVO');


% Parallel Processing
if matlabpool('size') == 0 % checking to see if my pool is already open
    matlabpool close force local
    matlabpool open local 4 
end

for topo = 1:length(possible_topo)
    fprintf('Topo: %i - Perform NN Train\n', possible_topo(topo));
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
        
        %if develop_mode, data_norm = data_norm(1:10,:);end
        %Train a NN
        parfor i_init = 1:n_init
            if (develop_mode) && (i_init > 2), continue; end
            fprintf('Topo: %i - iFold: %i of %i - Init: %i of %i\n',possible_topo(topo), ifolds, n_folds, i_init, n_init);
            [trained_nn{i_init}, train_description{i_init}] = train_neural_network(data_norm, target2train_norm, itrn, ival, itst, possible_topo(topo), train_fnc, perf_fnc, act_fnc, n_epochs, show);
        end
        save(sprintf('%s/mat/RunNNTrainCV_topo_%02i_fold_%02i.mat',outputpath,possible_topo(topo),ifolds),'trained_nn','train_description');
    end
end

if matlabpool('size') ~= 0 % checking to see if my pool is already open
    %matlabpool close
    matlabpool close force local
end