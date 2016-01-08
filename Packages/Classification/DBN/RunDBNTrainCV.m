clear all;
close all;

% Projeto Marinha do Brasil

% Autor: Lucas de Andrade Cerqueira
% Laboratorio de Processamento de Sinais - UFRJ


% Objetivo: Classificacao de navios utilizando Deep Belief Networks (DBN)

fprintf('Starting %s.m\n',mfilename('fullpath'));

% path for data
workspace = getenv('MARINHA_WORKSPACE');
datapath = getenv('OUTPUTDATAPATH');
outputpath = sprintf('%s/NoveltyDetection/DNN',datapath);

% Add functions Folder
addpath (genpath (sprintf ('%s/Packages/NoveltyDetection/functions', workspace)));

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

% create data struct
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

% Train NN procedure

n_folds = 2;

dbn_sizes = [100:100:1000];

CVO = cvpartition(target2train,'Kfold',n_folds);
normalization = 'mapminmax';
%save(sprintf('%s/TrainInformationDBN.mat', trainFolderPath),'normalization','CVO');

for ifolds = 1%:n_folds
    
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
    
 	train_x = ((data_norm(:,itrn) + 1)./2)';
 	test_x  = ((data_norm(:,itst) + 1)./2)';


	%train_y = double(train_y);
	%test_y  = double(test_y);
    
 	train_y = target2train_norm(:,itrn)';
 	test_y  = target2train_norm(:,itst)';

    erros = [];
    
	% Train a hidden unit DBN and use its weights to initialize a NN
    
    for i=1:length(dbn_sizes)
        
        fprintf ('Training DBN of size %i\n', dbn_sizes(i));
        
        rand('state',0)
        %train dbn
        dbn.sizes = [dbn_sizes(i)];
        opts.numepochs =   3;
        %opts.batchsize =    10;
        %opts.batchsize =    size (train_x,1);
        opts.batchsize =    8;
        opts.momentum  =    0.6;
        opts.alpha     =    0.01;
        dbn = dbnsetup(dbn, train_x, opts);
        dbn = dbntrain(dbn, train_x, opts);

        % Unfold DBN to NN
        nn = dbnunfoldtonn(dbn, 4);
        nn.activation_function = 'sigm';
        nn.learningRate = 1;

        %train nn
        opts.numepochs =  20;
        %opts.batchsize = size (train_x,1);
        opts.batchsize = 8;
        nn = nntrain(nn, train_x, train_y, opts, test_x, test_y);
        [er, bad] = nntest(nn, test_x, test_y);
        [a,b,c,d] = confusion(target2train_norm(:,itst)',sim_nn(nn,data_norm(:,itst)'));
        
        erros = [erros er];
    end
end
