% Projeto Marinha do Brasil

% Autor: Natanael Junior (natmourajr@gmail.com)
% Laboratorio de Processamento de Sinais - UFRJ


% Objetivos: Treinar (com Validacao Cruzada) redes neurais para classificacao de navios

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

data2train = [];
target2train = [];
target2train_norm = [];

% create data struct
for novelty_class = 1:numel(class_labels) % todas as classes
    
    if(exist(sprintf('%s/TrainInformationNN_novelty_%s.mat',outputpath,class_labels{novelty_class}),'file')),
        fprintf('The File %s already exists!!!!\n',sprintf('%s/TrainInformationDNN_novelty_%s.mat',outputpath,class_labels{novelty_class}));
        continue;
    end
    
    fprintf('Novelty Class: %s\n',class_labels{novelty_class});
    fprintf('Creating Data to Train\n');
    data2train = [];
    target2train = [];
    target2train_norm = [];
    
    for class = 1:numel(class_labels) % todas as classes
        
        if class == novelty_class, continue, end;
        
        % data
        aux = data_lofar.(class_labels{class});
        data2train   = [data2train aux];
        
        % target
        target2train = [target2train class*ones(1,length(aux))]; % numeric target
        
        aux = zeros(numel(class_labels)-1,length(aux)); % all class as 0
        
        if class < novelty_class
            aux(class,:) = 1*ones(1,length(aux)); % only one class as 1
        else
            aux(class-1,:) = 1*ones(1,length(aux)); % only one class as 1
        end
        
        target2train_norm = [target2train_norm aux];
    end
    
    CVO = cvpartition(target2train,'Kfold',n_folds);
    
    possible_topo = 2:3;
    
    % training parameters
    nn_epochs = 500;
    rbm_epochs = 100;
    normalization = 'mapminmax';
    save(sprintf('%s/TrainInformationDNN_novelty_%s.mat',outputpath,class_labels{novelty_class}),'data2train','target2train','target2train_norm','nn_epochs','rbm_epochs','normalization','CVO');
end


for novelty_class = 1%:numel(class_labels) % todas as classes
    fprintf('Novelty Class: %s\n',class_labels{novelty_class});
    fprintf('DNN Train Procedure\n');
    
    load(sprintf('%s/TrainInformationDNN_novelty_%s.mat',outputpath,class_labels{novelty_class}));

    for ifolds = 1%:n_folds
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
        sae = saesetup([size(data_norm,1) 200 100 50 25 10]);
        sae.ae{1}.activation_function       = 'sigm';
        sae.ae{1}.learningRate              = 1;
        sae.ae{1}.inputZeroMaskedFraction   = 0.5;
        opts.numepochs =   rbm_epochs;
        opts.batchsize = length(itrn);
        sae = saetrain(sae, data_norm(:,itrn)', opts);

        rbm{ifolds} = sae;
        
        nn_representation = nnsetup([size(data_norm,1) 200 100 50 25 10]);  
        nn_representation.activation_function              = 'sigm';
        nn_representation.learningRate                     = 0.9;
        nn_representation.W{1} = sae.ae{1}.W{1};
        nn_representation.W{2} = sae.ae{2}.W{1};
        nn_representation.W{3} = sae.ae{3}.W{1};
        nn_representation.W{4} = sae.ae{4}.W{1};
        nn_representation.W{5} = sae.ae{5}.W{1};
        
        save(sprintf('%s/mat/RunDNNTrainCV_NN-Representation_novelty_%s.mat',outputpath,class_labels{novelty_class}),'nn_representation');
                
%%%%%%%%%%%%%%%%%%%%%%%%
        return;
%%%%%%%%%%%%%%%%%%%%%%%%
        
        %Mudei a ultima camada pra 3
        nn = nnsetup([size(data_norm,1) 100 100 10 3]);  
        nn.activation_function              = 'sigm';
        nn.learningRate                     = 0.9;
        nn.W{1} = sae.ae{1}.W{1};
        nn.W{2} = sae.ae{2}.W{1};
        nn.W{3} = sae.ae{3}.W{1};
        

        opts.numepochs =   nn_epochs;
        opts.batchsize = length(itrn)/3;
        %warning off;
        warning ('off', 'all');
        nn = nntrain(nn, data_norm(:,itrn)', target2train_norm(:,itrn)', opts);
        %warning on;
        warning ('on', 'all');

        dnn{ifolds} = nn;

    end
    
    save(sprintf('%s/mat/RunDNNTrainCV_novelty_%s.mat',outputpath,class_labels{novelty_class}),'dnn','rbm');
end
    


return;
