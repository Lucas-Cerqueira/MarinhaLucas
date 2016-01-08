% Projeto Marinha do Brasil

% Autor: Natanael Junior (natmourajr@gmail.com)
% Laboratorio de Processamento de Sinais - UFRJ


% Objetivos: Treinar (com Validacao Cruzada) redes neurais para deteccao de novidade em sinais proveniente de navios

fprintf('Starting %s.m\n',mfilename('fullpath'));

% path for data
workspace = getenv('MARINHA_WORKSPACE');
datapath = getenv('OUTPUTDATAPATH');
outputpath = sprintf('%s/NoveltyDetection/DNN',datapath);

% Add functions Folder
addpath (genpath (sprintf ('%s/Packages/NoveltyDetection/functions', workspace)));

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

% create data struct
for novelty_class = 1:numel(class_labels) % todas as classes
    
%     if(exist(sprintf('%s/TrainInformationDNN_NN_novelty_%s.mat',outputpath,class_labels{novelty_class}),'file')),
%         fprintf('The File %s already exists!!!!\n',sprintf('%s/TrainInformationDNN_NN_novelty_%s.mat',outputpath,class_labels{novelty_class}));
%         continue;
%     end
    
    fprintf('Novelty Class: %s\n',class_labels{novelty_class});
    fprintf('Creating Data to Train\n');
    data2train = [];
    target2train = [];
    target2train_norm = [];
    
    for class = 1:numel(class_labels) % todas as classes
        fprintf('Processing class: %s\n',class_labels{class})
        % data
        if class == novelty_class, continue, end;
        
        % data
        aux = data_lofar.(class_labels{class});
        data2train   = [data2train aux];
        
        % target
        target2train = [target2train class*ones(1,length(aux))]; % numeric target
        
        aux = -1*ones(numel(class_labels)-1,length(aux)); % all class as -1
        
        if class < novelty_class
            aux(class,:) = 1*ones(1,length(aux)); % only one class as 1
        else
            aux(class-1,:) = 1*ones(1,length(aux)); % only one class as 1
        end
        
        target2train_norm = [target2train_norm aux];

        
    end
    
    CVO = cvpartition(target2train,'Kfold',n_folds);
    
    possible_topo = 10;
    
    % training parameters
    train_fnc = 'trainlm'; % weights update function
    perf_fnc = 'mse'; % error function
    act_fnc = {'tansig' 'tansig'}; % activation function
    n_epochs = 500;
    show = true;
    normalization = 'mapminmax';
    save(sprintf('%s/TrainInformationDNN_NN_novelty_%s.mat',outputpath,class_labels{novelty_class}),'data2train','target2train','target2train_norm','possible_topo','train_fnc','perf_fnc','act_fnc','n_epochs','normalization','CVO');
end

% Train NN procedure

result_norm = [];
novelty_result_norm = [];

for novelty_class = 1:numel(class_labels) % todas as classes
    fprintf('Novelty Class: %s\n',class_labels{novelty_class});
    fprintf('Train Procedure\n');
    
    load(sprintf('%s/TrainInformationDNN_NN_novelty_%s.mat',outputpath,class_labels{novelty_class}));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            load (sprintf('%s/mat/RunDNNTrainCV_AfterTraining_NN-Representation_novelty_%s.mat',outputpath,class_labels{novelty_class}));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for topo = 1:length(possible_topo)
        fprintf('Topo: %i\n', possible_topo(topo));
        for ifolds = 1:n_folds
            trained_nn = [];
            train_description = [];
            
            fprintf('Fold: %i of %i\n',ifolds, n_folds);
    
%             if (develop_mode) && (ifolds > 2), break; end
%             trn_id =  CVO.training(ifolds);
%             tst_id =  CVO.test(ifolds);
%             
%             itrn = []; itst = [];
%             for i = 1:length(data2train)
%                 if trn_id(i) == 1
%                     itrn = [itrn;i];
%                 else
%                     itst = [itst;i];
%                 end
%             end
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
            
            novelty_data = data_lofar.(class_labels {novelty_class});
          
            
            result = sim_nn(nn_representation, data_norm');
            
%           result = sim_nn (nn_representation, data_norm(:,target2train~=novelty_class)');
            novelty_result = sim_nn(nn_representation,novelty_data');
            
%             [~,norm_fact_result] = mapminmax(result(itrn,:)');
%             result_norm {novelty_class} = mapminmax('apply',result',norm_fact_result);
%             novelty_result_norm {novelty_class} = mapminmax('apply', novelty_result',norm_fact_result);
            
            %Train a NN
            for i_init = 1:n_init
                if (develop_mode) && (i_init > 2), break; end
                fprintf('Topo: %i - iFold: %i of %i - Init: %i of %i\n',possible_topo(topo), ifolds, n_folds, i_init, n_init);
                [trained_nn{i_init}, train_description{i_init}] = train_neural_network(result', target2train_norm, itrn, ival, itst, possible_topo(topo), train_fnc, perf_fnc, act_fnc, n_epochs, show);
            end
            save(sprintf('%s/mat/RunDNN_NNTrainCV_novelty_%s_topo_%02i_fold_%02i.mat',outputpath,class_labels{novelty_class},possible_topo(topo),ifolds),'trained_nn','train_description','result','novelty_result');
        end
    end
    
end

return;
