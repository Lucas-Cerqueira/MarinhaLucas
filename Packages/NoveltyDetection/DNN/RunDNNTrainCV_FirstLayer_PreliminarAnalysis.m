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
    nn_epochs = 300;
    rbm_epochs = 300;
    normalization = 'mapminmax';
    save(sprintf('%s/TrainInformationDNN_novelty_%s.mat',outputpath,class_labels{novelty_class}),'data2train','target2train','target2train_norm','nn_epochs','rbm_epochs','normalization','CVO');
end



for novelty_class = 1:numel(class_labels) % todas as classes
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
        sae = saesetup([size(data_norm,1) 100 100]);
        sae.ae{1}.activation_function       = 'sigm';
        sae.ae{1}.learningRate              = 1;
        sae.ae{1}.inputZeroMaskedFraction   = 0.5;
        opts.numepochs =   rbm_epochs;
        opts.batchsize = length(itrn);
        sae = saetrain(sae, data_norm(:,itrn)', opts);

        rbm{ifolds} = sae;
        
        %Cria uma rede neural só com as primeira camada de representação da DNN
        nn_representation = nnsetup([size(data_norm,1) 100]);  
        nn_representation.activation_function              = 'sigm';
        nn_representation.learningRate                     = 1;
        nn_representation.W{1} = sae.ae{1}.W{1};     

        save (sprintf('%s/mat/RunDNNTrainCV_NN-Representation_novelty_%s.mat',outputpath,class_labels{novelty_class}),'nn_representation', 'itrn', 'itst')
        
        
        %Plota os gráficos da saída da primeira camada antes do treinamento
        %de classificação
        
        novelty_data = data_lofar.(class_labels {novelty_class});

        result = sim_nn(nn_representation,data_norm(:,target2train~=novelty_class)');
        novelty_result = sim_nn(nn_representation,novelty_data');
        
        [~, result_norm_fact] = mapminmax(result(itrn,:)');
        
        result_norm = mapminmax('apply', result' ,result_norm_fact);
        novelty_result_norm = mapminmax('apply', novelty_result' ,result_norm_fact);
        
        v_colors = [0 0 1; 1 0 0; 0 0.5 0; 0.75 0.75 0];
        
        h = figure;
        for i=1:16
            subplot (4,4,i);
            hold on;
            for class = 1:numel(class_labels)
                if class == novelty_class
                    [y,x] = hist (novelty_result_norm(i,:), 50);
                    plot (x, y, 'k-', 'linewidth', 2.0);
                else
                    [y,x] = hist (result_norm (i, target2train==class), 50);
                    plot (x, y, '-', 'color', v_colors (class,:), 'linewidth', 2.0);
                end
                %xlim([0 1]);
            end
            hold off;
            legend ('Novelty');
        end
        
        saveas(h,sprintf ('%s/pict/result_norm_firstLayerBeforeTraining_novelty_representation_%s.png', outputpath, class_labels{novelty_class}));
        
%         h = figure;
%         for i=1:16
%             subplot (4,4,i);
%             hold on;
%             for class = 1:numel(class_labels)
%                 if class == novelty_class
%                     [y,x] = hist (novelty_result(:, i)', 50);
%                     plot (x, y, 'k-', 'linewidth', 2.0);
%                 else
%                     [y,x] = hist (result (target2train==class, i)', 50);
%                     plot (x, y, '-', 'color', v_colors (class,:), 'linewidth', 2.0);
%                 end
%                 %xlim([0 1]);
%             end
%             hold off;
%         end
%         
%         saveas(h,sprintf ('%s/pict/result_firstLayerBeforeTraining_novelty_representation_%s.png', outputpath, class_labels{novelty_class}));

        
        %Treinamento da rede neural para classificação
        nn = nnsetup([size(data_norm,1) 100 100 3]);  
        nn.activation_function              = 'sigm';
        nn.learningRate                     = 0.9;
        nn.W{1} = sae.ae{1}.W{1};
        nn.W{2} = sae.ae{2}.W{1};
        %nn.W{3} = sae.ae{3}.W{1};
        %nn.W{4} = sae.ae{4}.W{1};
        opts.numepochs =   nn_epochs;
        opts.batchsize = length(itrn)/3;
        warning off;
        %warning ('off', 'all');
        nn = nntrain(nn, data_norm(:,itrn)', target2train_norm(:,itrn)', opts);
        warning on;
        %warning ('on', 'all');

        nn_representation = nnsetup([size(data_norm,1) 100]);  
        nn_representation.activation_function              = 'sigm';
        nn_representation.learningRate                     = 1;
        nn_representation.W{1} = nn.W{1};
        
        %save (sprintf('%s/mat/RunDNNTrainCV_NN-Representation_novelty_%s.mat',outputpath,class_labels{novelty_class}),'nn_representation', 'itrn', 'itst');

        
        result = sim_nn(nn_representation,data_norm(:,target2train~=novelty_class)');
        novelty_result = sim_nn(nn_representation,novelty_data');
        
        [~, result_norm_fact] = mapminmax(result(itrn,:)');
        
        result_norm = mapminmax('apply', result' ,result_norm_fact);
        novelty_result_norm = mapminmax('apply', novelty_result' ,result_norm_fact);
       
        h = figure;
        for i=1:16
            subplot (4,4,i);
            hold on;
            for class = 1:numel(class_labels)
                if class == novelty_class
                    [y,x] = hist (novelty_result_norm(i,:), 50);
                    plot (x, y, 'k-', 'linewidth', 2.0);
                else
                    [y,x] = hist (result_norm (i, target2train==class), 50);
                    plot (x, y, '-', 'color', v_colors (class,:), 'linewidth', 2.0);
                end
                %xlim([0 1]);
            end
            hold off;
        end
        
        saveas(h,sprintf ('%s/pict/result_norm_firstLayerAfterTraining_novelty_representation_%s.png', outputpath, class_labels{novelty_class}));
        
%         h = figure;
%         for i=1:16
%             subplot (4,4,i);
%             hold on;
%             for class = 1:numel(class_labels)
%                 if class == novelty_class
%                     [y,x] = hist (novelty_result(:, i)', 50);
%                     plot (x, y, 'k-', 'linewidth', 2.0);
%                 else
%                     [y,x] = hist (result (target2train==class, i)', 50);
%                     plot (x, y, '-', 'color', v_colors (class,:), 'linewidth', 2.0);
%                 end
%                 %xlim([0 1]);
%             end
%             hold off;
%         end
%         
%         saveas(h,sprintf ('%s/pict/result_firstLayerAfterTraining_novelty_representation_%s.png', outputpath, class_labels{novelty_class}));
        
        
        
%         result = sim_nn (nn,data_norm(:,target2train~=novelty_class)');
%         novelty_result = sim_nn(nn,novelty_data');
%      
%         
%         h = figure;
%         for class=1:numel(class_labels)
%             %hold on;
%             for i = 1:3
%                 subplot (numel(class_labels),3, 3*(class-1) + i);
%                 
%                 if class == novelty_class
%                     [y,x] = hist (novelty_result(:,i)', 50);
%                     plot (x, y, 'k-', 'linewidth', 2.0);
%                 else
%                     [y,x] = hist (result (target2train==class, i)', 50);
%                     plot (x, y, '-', 'color', v_colors (class,:), 'linewidth', 2.0);
%                 end
%                 xlim([0 1]);
%             end
%             %hold off;
%         end
%         saveas(h,sprintf ('result_preliminar_novelty_classification_%s.png', class_labels{novelty_class}));
        
    end
    %save(sprintf('%s/mat/RunDNNTrainCV_novelty_%s.mat',outputpath,class_labels{novelty_class}),'dnn','rbm');
end
  

return;