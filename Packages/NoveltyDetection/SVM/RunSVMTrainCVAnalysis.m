% Projeto Marinha do Brasil

% Autor: Natanael Junior (natmourajr@gmail.com)
% Laboratorio de Processamento de Sinais - UFRJ


% Objetivos: Analisar (com Validacao Cruzada) SVM para deteccao de novidade em sinais proveniente de navios

fprintf('Starting %s.m\n',mfilename('fullpath'));

clear all;
close all;

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

% Analysis
for novelty_class = 1:numel(class_labels) % todas as classes
    
    if(~exist(sprintf('%s/TrainInformationSVM_novelty_%s.mat',outputpath,class_labels{novelty_class}),'file')),
        fprintf('No train process detected!!!\n');
        continue;
    end
    
    load(sprintf('%s/TrainInformationSVM_novelty_%s.mat',outputpath,class_labels{novelty_class}));
    
    fprintf('Novelty: %s - Analysing\n',class_labels{novelty_class});
    
    for ifolds = 1:n_folds
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
        
        
        load(sprintf('%s/mat/RunSVMTrainCV_novelty_%s_classifiertype_%s_fold_%02i.mat',outputpath,class_labels{novelty_class},classifier_type,ifolds));
        
        data_norm = (base(:,1:cut_index)/norm(base(:,1:cut_index)))'*data_norm;
                
        
        
        train_data = gendatoc(data_norm(:,(trn_id') & target2train ~= novelty_class)',data_norm(:,(trn_id') & target2train == novelty_class)');
        test_data = gendatoc(data_norm',[]);
        
        
        svm_result = test_data*single_class_classifier_for_all_known_classes*labeld;
        
        
        svm_num_result = [];
        
        for event = 1:length(svm_result)
            svm_num_result(event) = strcmp(svm_result(event,:),'target ');
        end
        
        svm_target = target2train ~= novelty_class;
        
        [SP(ifolds),Eff(ifolds,:)] = computeSP(svm_target(tst_id),svm_num_result(tst_id));
        
        svm_num_result = [];
        
        for iclass = 1:numel(class_labels)
            if iclass == novelty_class, continue; end;
            svm_result = test_data*single_class_classifier_for_one_class{iclass - (iclass > novelty_class)}*labeld;
            
            for event = 1:length(svm_result)
                svm_num_result(iclass - (iclass > novelty_class),event) = strcmp(svm_result(event,:),'target ');
            end
        end
        
        %Compute results
        novelty_result = (svm_num_result(1,:)==0 & svm_num_result(2,:)==0 & svm_num_result(3,:)==0);
        
        for iclass = 1:numel(class_labels)
            if iclass == novelty_class, continue; end
            svm_target(iclass -(iclass > novelty_class),:) = target2train == iclass;
        end
        novelty_target = target2train == novelty_class;
        
        
        for iclass = 1:numel(class_labels) % known class against all others
            if iclass == novelty_class, continue; end
            [SP_class(iclass-(iclass > novelty_class), ifolds),Eff_class(iclass-(iclass > novelty_class),ifolds,:)] = computeSP(svm_target(iclass-(iclass > novelty_class),tst_id),svm_num_result(iclass-(iclass > novelty_class),tst_id));
        
        end
        
        [SP_novelty(ifolds),Eff_novelty(ifolds,:)] = computeSP(novelty_target(tst_id),novelty_result(tst_id));
        
    end
    
    fprintf('\nNovelty: %s\n',class_labels{novelty_class});
    fprintf('Known Class against Novelty\n');
    fprintf('SP: %1.3f +- %1.6f\n',mean(SP), 100*var(SP));
    fprintf('Eff[Unknown Class]: %1.3f +- %1.6f\n',mean(Eff(:,1)), 100*var(Eff(:,1)));
    fprintf('Eff[Known Class]: %1.3f +- %1.6f\n\n',mean(Eff(:,2)), 100*var(Eff(:,2)));
    
    fprintf('One Class against others\n');
    mean_SP = mean(SP_class,2);
    var_SP  = var(SP_class,[],2);
    
    mean_Eff = mean(Eff_class,2);
    var_Eff = var(Eff_class,[],2);
    
    str_aux = {'Class', 'No Class'};
    
    for iclass = 1:numel(class_labels)
        if iclass ~= novelty_class
            fprintf('SP Class-NoClass(%s): %1.3f +- %1.6f\n',class_labels{iclass},mean_SP(iclass-(iclass > novelty_class)), 100*var_SP(iclass-(iclass > novelty_class)));
            fprintf('Eff_Class[%s]: %1.3f +- %1.6f\n',class_labels{iclass},mean_Eff(iclass-(iclass > novelty_class),2), 100*var_Eff(iclass-(iclass > novelty_class),2));
            fprintf('Eff_NoClass[%s]: %1.3f +- %1.6f\n',class_labels{iclass},mean_Eff(iclass-(iclass > novelty_class),1), 100*var_Eff(iclass-(iclass > novelty_class),1));
        else
            fprintf('SP(Novelty): %1.3f +- %1.6f\n',mean(SP_novelty), 100*var(SP_novelty));
            mean_Eff_novelty = mean(Eff_novelty,1);
            var_Eff_novelty = var(Eff_novelty,[],1);
            fprintf('Eff_Class[%s]: %1.3f +- %1.6f\n','Novelty Class',mean_Eff_novelty(:,2), 100*var_Eff_novelty(:,2));
            fprintf('Eff_NoClass[%s]: %1.3f +- %1.6f\n','Novelty Class',mean_Eff_novelty(:,1), 100*var_Eff_novelty(:,1));
        end
    end
end


