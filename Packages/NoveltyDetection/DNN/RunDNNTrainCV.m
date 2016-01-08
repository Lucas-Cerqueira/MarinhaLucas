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
    
    % training parameters
    nn_epochs = 200;
    rbm_epochs = 200;
    normalization = 'mapminmax';

    %If the DNN will be trained for classification or not
    classificationTraining = false;

    %Keep DNN representation training weights or let it be tuned
    keepAEWeights = false;

    %DNN Dimensions
    topos = [[size(data2train,1) 200 100];
             [size(data2train,1) 100 25];
             [size(data2train,1) 40 4]];
         
    
    topos = [size(data2train,1) 40 4];

    save(sprintf('%s/TrainInformationDNN_novelty_%s.mat',outputpath,class_labels{novelty_class}),'data2train','target2train','target2train_norm','nn_epochs','rbm_epochs','normalization','CVO', 'classificationTraining','keepAEWeights','topos');
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
        
        for indexTopo=1%:size(topos,2)
        
            %Train a DNN
            sae = saesetup(topos(indexTopo,:));
            sae.ae{1}.activation_function       = 'sigm';
            sae.ae{1}.learningRate              = 1;
            sae.ae{1}.inputZeroMaskedFraction   = 0.5;
            opts.numepochs =   rbm_epochs;
            opts.batchsize = length(itrn);
            sae = saetrain(sae, data_norm(:,itrn)', opts);

            rbm{ifolds} = sae;

            %DNN classification training
            if (classificationTraining)
                if (keepAEWeights)
                    %Create a NN with 2 layers: the representation last layer and an output layer
                    dnn_dimension = [topos(indexTopo,end) 3];
                    nn = nnsetup(dnn_dimension);  
                    nn.activation_function              = 'sigm';
                    nn.learningRate                     = 0.9;

                    %Create a auxiliar DNN to evaluate the representation output
                    dnn_representation = nnsetup(dnn_dimension);  
                    dnn_representation.activation_function              = 'sigm';
                    dnn_representation.learningRate                     = 1;
                    for (index=1:length(dnn_dimension)-1)
                        dnn_representation.W{index} = sae.ae{index}.W{1};
                    end
                    result = sim_nn(dnn_representation,data_norm(:,target2train~=novelty_class)');
                    

                    opts.numepochs =   nn_epochs;
                    opts.batchsize = length(itrn)/3;
                    warning off;
                    nn = nntrain(nn, result, target2train_norm(:,itrn)', opts);
                    warning on;
                else
                    %Create a DNN loading the auto-encoder initialized weights and train it for classification
                    dnn_dimension = [topos(indexTopo,:) 3];
                    nn = nnsetup(dnn_dimension);
                    for (index=1:(length(dnn_dimension)-2))
                        dnn_representation.W{index} = sae.ae{index}.W{1};
                    end

                    nn.activation_function              = 'sigm';
                    nn.learningRate                     = 0.9;

                    opts.numepochs =   nn_epochs;
                    opts.batchsize = length(itrn)/3;
                    warning off;
                    nn = nntrain(nn, data_norm(:,itrn)', target2train_norm(:,itrn)', opts);
                    warning on;
                end

                
                dnn_representation = nn;

                %save (sprintf('%s/mat/RunDNNTrainCV_AfterTraining_NN-Representation_novelty_%s_fold_%i_topo_%i.mat',outputpath,class_labels{novelty_class},ifolds,indexTopo),'dnn_representation', 'itrn', 'itst');
            else
                %Cria uma rede neural só com as primeiras camadas da DNN, para
                %representação da entrada
                dnn_representation = nnsetup(topos(indexTopo,:));  
                dnn_representation.activation_function              = 'sigm';
                dnn_representation.learningRate                     = 1;
                for (index=1:length(topos(indexTopo,:))-1)
                    dnn_representation.W{index} = sae.ae{index}.W{1};
                end

                %save (sprintf('%s/mat/RunDNNTrainCV_BeforeTraining_NN-Representation_novelty_%s_fold_%i_topo_%i.mat',outputpath,class_labels{novelty_class},ifolds,indexTopo),'dnn_representation', 'itrn', 'itst');
            end
        
            save (sprintf('%s/mat/RunDNNTrainCV_DNN-Representation_novelty_%s_fold_%i_topo_%i.mat',outputpath,class_labels{novelty_class},ifolds,indexTopo),'dnn_representation', 'itrn', 'itst', 'rbm');
            
        end
    end
end

            %Iterates analysing the output of each layer
    %         for (layerIndex=2:length(dnn_dimension))
    %             
    %             nn_auxiliar = nnsetup (dnn_dimension(1:layerIndex));
    %             nn_auxiliar.activation_function              = 'sigm';
    %             nn_auxiliar.learningRate                     = 1;
    %             for (index=1:layerIndex-1)
    %                 nn_auxiliar.W{index} = dnn_representation.W{index};
    %             end
    %             
    %             
    %             novelty_data = data_lofar.(class_labels {novelty_class});
    % 
    %             result = sim_nn(nn_auxiliar,data_norm(:,target2train~=novelty_class)');
    %             novelty_result = sim_nn(nn_auxiliar,novelty_data');
    % 
    %     %         if strcmp(normalization, 'mapstd') == 1
    %     %             % Mean = 0 and var = 1
    %     %             [~, result_norm_fact] = mapstd(result(itrn,:)');
    %     %             result_norm = mapstd('apply', result' ,result_norm_fact);
    %     %             novelty_result_norm = mapstd('apply', novelty_result' ,result_norm_fact);
    %     %         else
    %     %             % Min = -1 and Max = 1
    %     %             [~, result_norm_fact] = mapminmax(result(itrn,:)');
    %     %             result_norm = mapminmax('apply', result' ,result_norm_fact);
    %     %             novelty_result_norm = mapminmax('apply', novelty_result' ,result_norm_fact);
    %     %         end
    % 
    % 
    %             v_colors = [0 0 1; 1 0 0; 0 0.5 0; 0.75 0.75 0];
    % 
    %     %         h = figure;
    %     %         for i=1:10
    %     %             subplot (2,5,i);
    %     %             hold on;
    %     %             for class = 1:numel(class_labels)
    %     %                 if class == novelty_class
    %     %                     [y,x] = hist (novelty_result_norm(i,:), 50);
    %     %                     plot (x, y, 'k-', 'linewidth', 2.0);
    %     %                 else
    %     %                     [y,x] = hist (result_norm (i, target2train==class), 50);
    %     %                     plot (x, y, '-', 'color', v_colors (class,:), 'linewidth', 2.0);
    %     %                 end
    %     %                 %xlim([0 1]);
    %     %             end
    %     %             hold off;
    %     %         end
    %     %         
    %     %         saveas(h,sprintf ('result_norm_preliminar_novelty_representation_%s.png', class_labels{novelty_class}));
    % 
    %             %Check if this is the DNN's last layer
    %             if (layerIndex == length (dnn_dimension))%-1) %% Before the last layer: 10 neurons
    %                 h = figure;
    %                 %%%%%
    %                 hold on;
    %                 title(sprintf('SAE Last Layer Representation (Novelty: %s)',class_labels{novelty_class}),'FontSize', 15,'FontWeight', 'bold');
    %                 hold off;
    %                 %%%%%%
    %                 for i=1:10
    %                     subplot (2,5,i);
    %                     hold on;
    %                     
    %                     %%%%%%%%%
    %                     xlabel(sprintf('Output'),'FontSize', 8,'FontWeight', 'bold');
    %                     ylabel(sprintf('Propability'),'FontSize', 8,'FontWeight', 'bold');
    %                     %%%%%%%%%
    % 
    %                     for class = 1:numel(class_labels)
    %                         if class == novelty_class
    %                             [y,x] = hist (novelty_result(:, i)', 50);
    %                             %plot (x, y/sum(y*diff(x(1:2))), 'k-', 'linewidth', 2.0);
    %                             plot (x, y/max(y), 'k-', 'linewidth', 2.0);
    %                         else
    %                             [y,x] = hist (result (target2train==class, i)', 50);
    %                             %plot (x, y/sum(y*diff(x(1:2))), '-', 'color', v_colors (class,:), 'linewidth', 2.0);
    %                             plot (x, y/max(y), '-', 'color', v_colors (class,:), 'linewidth', 2.0);
    %                         end
    %                         %xlim([0 1]);
    %                         %ylim([0 1]);
    %                     end
    %                     hold off;
    %                 end
    %                 
    %             else
    %                 if (layerIndex == length (dnn_dimension))  %%Last layer: 3 neurons
    %                     h = figure;
    %                     hold on;
    %                     for classIndex=1:numel(class_labels)
    %                         for neuron=1:3
    %                             subplot (numel(class_labels), 3, 3*(classIndex-1) + neuron)
    %                             if classIndex == novelty_class
    %                                 [y,x] = hist (novelty_result(:, neuron)', 50);
    %                                 plot (x, y, 'k-', 'linewidth', 2.0);
    %                             else
    %                                 [y,x] = hist (result (target2train==classIndex, neuron)', 50);
    %                                 plot (x, y, '-', 'color', v_colors (classIndex,:), 'linewidth', 2.0);
    %                             end
    %                         xlim([0 1]);    
    %                         end
    %                     end
    %                 hold off;
    %                 else
    %                     h = figure;
    %                     for i=1:20
    %                         subplot (4,5,i);
    %                         hold on;
    %                         for class = 1:numel(class_labels)
    %                             if class == novelty_class
    %                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        [y,x] = hist (novelty_result(:, i)', 50);
    %                                 plot (x, y, 'k-', 'linewidth', 2.0);
    %                             else
    %                                 [y,x] = hist (result (target2train==class, i)', 50);
    %                                 plot (x, y, '-', 'color', v_colors (class,:), 'linewidth', 2.0);
    %                             end
    %                             %xlim([0 1]);
    %                         end
    %                         hold off;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    %                     end
    %                 end
    %             end
 

return;
