% Projeto Marinha do Brasil

% Autor: Natanael Junior (natmourajr@gmail.com)
% Laboratorio de Processamento de Sinais - UFRJ


% Objetivos: Treinar (com Validacao Cruzada) SVM para deteccao de novidade em sinais proveniente de navios

fprintf('Starting %s.m\n',mfilename('fullpath'));

close all;
clear all;

% path for data
workspace = getenv('MARINHA_WORKSPACE');
datapath = getenv('OUTPUTDATAPATH');
outputpath = sprintf('%s/NoveltyDetection/SVM',datapath);

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

% create data struct
for novelty_class = 1:numel(class_labels) % todas as classes
    
    if(exist(sprintf('%s/TrainInformationSVM_novelty_%s.mat',outputpath,class_labels{novelty_class}),'file')),
        fprintf('The Train Information File %s already exists!!!!\n',sprintf('%s/TrainInformationSVM_novelty_%s.mat',outputpath,class_labels{novelty_class}));
        continue;
    end
    
    fprintf('Novelty Class: %s\n',class_labels{novelty_class});
    fprintf('Creating Train Information File\n');
    data2train = [];
    target2train = [];
    target2train_norm = [];
    
    for class = 1:numel(class_labels) % todas as classes
        % data
        aux = data_lofar.(class_labels{class});
        data2train   = [data2train aux];
        
        % target
        target2train = [target2train class*ones(1,length(aux))]; % numeric target
        
    end
    
    %train parameters
    normalization = 'mapstd';
    classifier_type = 'svm';
    rejection_factor = 0.01;
    
    CVO = cvpartition(target2train,'Kfold',n_folds);
    save(sprintf('%s/TrainInformationSVM_novelty_%s.mat',outputpath,class_labels{novelty_class}),'data2train','target2train','CVO', 'classifier_type','rejection_factor','normalization');
end

% Train Process

for novelty_class = 1:numel(class_labels) % todas as classes
    fprintf('Novelty Class: %s\n',class_labels{novelty_class});
    fprintf('Train Procedure\n');
    
    load(sprintf('%s/TrainInformationSVM_novelty_%s.mat',outputpath,class_labels{novelty_class}));
    
    for ifolds = 1:n_folds
        single_class_classifier = [];
        
        fprintf('Fold: %i of %i\n',ifolds, n_folds);
        
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
        
        %Train a Single Class Classifier
        
        rmpath(genpath('../functions'));
        % PCA Compression
        [~,base,energy] = pca(data_norm(:,itrn));
        addpath(genpath('../functions'));
        
        energy = 100*energy/sum(energy);
        
        total_of_energy = 0;
        cut_index = 0;
        for ienergy = 1:length(energy)
            total_of_energy = total_of_energy + energy(ienergy);
            if total_of_energy >= 80.0
                cut_index = ienergy;
                break;
            end
        end
        
        cut_index = 10;
        data_norm = (base(:,1:cut_index)/norm(base(:,1:cut_index)))'*data_norm;
        
        train_data = gendatoc(data_norm(:,(trn_id') & target2train ~= novelty_class)',data_norm(:,(trn_id') & target2train == novelty_class)');
        
        single_class_classifier_for_all_known_classes = parzen_dd(train_data,1*rejection_factor,[]); % classifier types
        %kernel_matrix = dd_proxm([],'r',3); % Explicit RBF kernel with sigma=3
        %single_class_classifier_for_all_known_classes = ksvdd(train_data,1*rejection_factor,kernel_matrix); % classifier types
        
        e = dd_error(train_data,single_class_classifier_for_all_known_classes);
        fprintf('\nSVM Train Proccess: All class against Novelty: %s - Error %1.5f%%\n', class_labels{novelty_class},100*e(1));
        
        for iclass = 1:numel(class_labels)
            if iclass == novelty_class, continue; end;
            train_data = gendatoc(data_norm(:,(trn_id') & target2train == iclass)',data_norm(:,(trn_id') & target2train ~= novelty_class & target2train ~= iclass)');
            
            single_class_classifier_for_one_class{iclass - (iclass > novelty_class)} = parzen_dd(train_data,rejection_factor,[]); % classifier types
            e = dd_error(train_data,single_class_classifier_for_one_class{iclass - (iclass > novelty_class)} );
            fprintf('\nSVM Train Proccess: One Class: %s - Error %1.5f%%\n', class_labels{iclass},100*e(1));
        end
        
        
        %single_class_classifier = incsvdd(train_data,10*rejection_factor,'r',0.1); % classifier types
        %single_class_classifier = gauss_dd(train_data,rejection_factor,0.1); % classifier types
        %single_class_classifier = svdd(train_data,10*rejection_factor,5); % classifier types
        %single_class_classifier = setlabels(single_class_classifier,{'  Known Class';'Unknown Class'});
        
        save(sprintf('%s/mat/RunSVMTrainCV_novelty_%s_classifiertype_%s_fold_%02i.mat',outputpath,class_labels{novelty_class},classifier_type,ifolds),'single_class_classifier_for_all_known_classes','single_class_classifier_for_one_class','cut_index');
        
        % export one picture
        if cut_index == 2
            v_colors = [0 0 1; 1 0 0; 0 0.5 0; 0.75 0.75 0];
            pict_handler = figure;
            
            v_marker = ['o' '+'];
            
            % plotting data
            for iclass = 1:numel(class_labels)
                fprintf('Plotting %s\n',class_labels{iclass});
                plot(data_norm(1,target2train == iclass),data_norm(2,target2train == iclass),v_marker(1+(iclass==novelty_class)),'Color',v_colors(iclass,:),'LineWidth',1.5);
                hold on;
            end
            H = class_labels;
            
            H{novelty_class} = 'Novelty';
            
            %plotting classifiers
            
            h = plotc(single_class_classifier_for_all_known_classes,'k');
            set(h,'Color',[0 0 0]);
            set(h,'LineStyle','-');
            set(h,'LineWidth',2.0);
            H{numel(class_labels)+1} = 'One-Class';
            
                       
            for iclass = 1:numel(class_labels)
                if iclass == novelty_class, continue; end;
                h = plotc(single_class_classifier_for_one_class{iclass - (iclass > novelty_class)},'k');
                set(h,'Color',v_colors(iclass,:));
                set(h,'LineStyle','--');
                set(h,'LineWidth',2.0);
                H{numel(class_labels)+1+iclass - (iclass > novelty_class)} = sprintf('%s Class.',class_labels{iclass});
            
            end
            
            hold off;
            grid on;
            
            h_legend = legend(H,'Location','SouthEast');
            
            title(sprintf('SVM: Scatter Plot (Novelty: %s)',class_labels{novelty_class}),'FontSize', 15,'FontWeight', 'bold');
            xlabel(sprintf('Component 1'),'FontSize', 15,'FontWeight', 'bold');
            ylabel(sprintf('Component 2'),'FontSize', 15,'FontWeight', 'bold');
            
            saveas(pict_handler,sprintf('%s/pict/RunSVMTrainCV_novelty_%s_classifiertype_%s_fold_%02i_scatter_train_plot.png',outputpath,class_labels{novelty_class},classifier_type,ifolds));
            close(pict_handler);
            
        end
    end
end

rmpath(genpath('../functions'));