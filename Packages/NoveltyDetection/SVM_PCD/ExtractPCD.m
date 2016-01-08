% Projeto de Classificacao para Marinha do Brasil

% Autor: Natanael Junior (natmourajr@gmail.com)
% Laboratorio de Processamento de Sinais - UFRJ


% Objetivos: Extrair PCDs do modo independente do banco de dados
%

% iniciando o script
clear all;
close all;
clc;

fprintf('Starting %s.m\n',mfilename('fullpath'));

% get path for data
outputpath = getenv('OUTPUTDATAPATH');


% importando funcoes
fprintf('Importing Functions\n');
addpath(genpath('../functions'));

% load data
fprintf('Load Data\n');

if(~exist(sprintf('%s/mat/raw_data.mat',outputpath),'file'))
    error('DO READ RAW DATA\n');
    exit;
else 
    load(sprintf('%s/mat/raw_data.mat',outputpath));
end


if(~exist(sprintf('%s/mat/lofar_data.mat',outputpath),'file'))
    fprintf('DO PERFORM_LOFAR_ANALYSIS\n');
    return;
end

load(sprintf('%s/mat/lofar_data.mat',outputpath)) 

fprintf('Mounting PCD Data\n');

data2pcd = [];
target2pcd = [];
target2pcd_norm = [];


for class = 1:numel(class_labels) % todas as classes
    fprintf('%s\n',class_labels{class});
    
    % data
    aux = data_lofar.(class_labels{class});
    data2pcd   = [data2pcd aux];

    % target
    target2pcd = [target2pcd class*ones(1,length(aux))]; % numeric target
  
    aux = -1*ones(numel(class_labels),length(aux)); % all class as -1
    aux(class,:) = 1*ones(1,length(aux)); % only one class as 1
    target2pcd_norm = [target2pcd_norm aux];
end

% Calculating PCD
fprintf('Calculating PCD\n');

n_folds = 10; n_init = 10; num_pcds = 20;
CVO = cvpartition(length(data2pcd),'Kfold',n_folds);
save(sprintf('%s/mat/pcd/pcd_k_fold_info.mat',outputpath),'n_folds','n_init','num_pcds','CVO');

for ifolds = 1:n_folds
    trn_id =  CVO.training(ifolds);
    tst_id =  CVO.test(ifolds);
    
    itrn = []; itst = [];
    for i = 1:length(data2pcd)
        if trn_id(i) == 1
            itrn = [itrn;i];
        else
            itst = [itst;i];
        end
    end
    
    % normalization
    [data_norm, norm_fact] = mapstd(data2pcd(:,itrn));
    data_norm = mapstd('apply', data2pcd ,norm_fact);
    
    trn_params.itrain = itrn;
    trn_params.itest = itst;
    trn_params.ivalid = itst;
    trn_params.train_fnc = 'trainlm';
    trn_params.perf_fnc = 'mse';
    trn_params.act_fnc = {'tansig' 'tansig'};
    trn_params.n_epochs = 1000;


    for i_init = 1:n_init
        fprintf('iFold: %i of %i - Init: %i of %i\n',ifolds, n_folds, i_init, n_init);
        
        exp_nn = [];
        trn_desc = [];
                
        for i_pcd = 1:num_pcds
            
            fprintf('Extracting PCD %i\n',i_pcd);
            % for each PCD train a NN with 1 neuron and save Weights.
            
            nn = newff(data_norm,target2pcd_norm,1,trn_params.act_fnc,trn_params.train_fnc);
            nn = init(nn);
            
            orth_data_norm = [];
            
            if i_pcd ~= 1
                
                data_proj_ant = zeros(size(data_norm));
                weig_proj_ant = zeros(1,size(data_norm,1));
                                               
                for j_pcd = 1:i_pcd-1
                    data_proj_ant = data_proj_ant + (((net{j_pcd}.IW{1}*data_norm)'*net{j_pcd}.IW{1})/(net{j_pcd}.IW{1}*net{j_pcd}.IW{1}'))'; % Gram–Schmidt process
                    weig_proj_ant = weig_proj_ant + (((net{j_pcd}.IW{1}*nn.IW{1}')*net{j_pcd}.IW{1})/(net{j_pcd}.IW{1}*net{j_pcd}.IW{1}')); % Gram–Schmidt process
                end
                
                % removing older directions
                orth_data_norm = data_norm - data_proj_ant;
                nn.IW{1} = nn.IW{1} - weig_proj_ant;
                 
            end
            
            %Definicoes de Treinamento.
            nn.trainParam.lr               = 0.000001;
            nn.trainParam.max_fail         = 0.5*trn_params.n_epochs;
            nn.trainParam.mc               = 0.99999;
            nn.trainParam.min_grad         = 1e-10;
            nn.trainParam.goal             = 0;
            nn.trainParam.epochs           = trn_params.n_epochs;
            nn.performFcn                  = trn_params.perf_fnc;
            nn.trainParam.show             = nn.trainParam.max_fail;
            nn.trainParam.showWindow       = false;
            nn.trainParam.showCommandLine  = true;
            
            nn.divideFcn                   = 'divideind';
            nn.divideParam.trainInd        = trn_params.itrain;
            nn.divideParam.testInd         = trn_params.itest;
            nn.divideParam.valInd          = trn_params.ivalid;
            
            %Treinamento da rede.
            if i_pcd == 1               
                [net{i_pcd}, trn_desc{i_pcd}]   = train(nn,data_norm,target2pcd_norm);
            else
                [net{i_pcd}, trn_desc{i_pcd}]   = train(nn,orth_data_norm,target2pcd_norm);
            end
        end
        save(sprintf('%s/mat/pcd/pcd_folds_%i_n_init_%i.mat',outputpath,ifolds,i_init),'net','trn_desc');
    end
end

% removendo funcoes
fprintf('Removing Functions\n');
rmpath(genpath('../functions'));

%exit;
