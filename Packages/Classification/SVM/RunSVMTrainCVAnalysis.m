% Projeto Marinha do Brasil

% Autor: Natanael Junior (natmourajr@gmail.com)
% Laboratorio de Processamento de Sinais - UFRJ


% Objetivos: Treinar (com Validacao Cruzada) redes neurais para classificacao de navios

fprintf('Starting %s.m\n',mfilename('fullpath'));

% path for data
workspace = getenv('MARINHA_WORKSPACE');
datapath = getenv('OUTPUTDATAPATH');
outputpath = sprintf('%s/Classification/SVM',datapath);

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

load(sprintf('%s/TrainInformationSVM.mat',outputpath));

SP = [];
Eff = [];

for ifolds = 1:n_folds
    if (develop_mode) && (ifolds > 2), continue; end
    fprintf('iFold: %i - Analysing SVM model\n',ifolds);
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
    
    load(sprintf('%s/mat/RunSVMTrainCV_fold_%02i.mat',outputpath,ifolds));
    
    svm_output = [];
    svm_target = [];
    
    for iclass = 1:numel(class_labels)
        svm_output(iclass,:) = svmclassify(svm_classifier{iclass},data_norm')';
        svm_target(iclass,:) = (target2train==iclass);
    end
    
    [SP(ifolds),Eff(ifolds,:)] = computeSP(svm_target,svm_output);
end


% compute sp mean and var
mean_sp = mean(SP);
var_sp = var(SP);
mean_eff = mean(Eff,1);
var_eff = var(Eff,[],1);


fprintf('\nSP: %1.5f%% +- %1.8f%%\n',mean_sp,var_sp);

for iclass = 1:numel(class_labels)
    fprintf('Eff[%s]: %1.5f%% +- %1.8f%%\n',class_labels{iclass},mean_eff(iclass),var_eff(iclass));
end

save(sprintf('%s/mat/RunSVMTrainCVAnalysis.mat',outputpath),'SP','Eff');

%show output dist per target

dist_v_colors = jet(n_folds);

h = figure;

for ifolds = 1:n_folds
    if (develop_mode) && (ifolds > 2), continue; end
    %fprintf('iFold: %i - Analysing SVM model\n',ifolds);
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
    
    load(sprintf('%s/mat/RunSVMTrainCV_fold_%02i.mat',outputpath,ifolds));
    
    svm_output = [];
    svm_target = [];
    
    for iclass = 1:numel(class_labels)
        svm_output(iclass,:) = svmclassify(svm_classifier{iclass},data_norm')';
        svm_target(iclass,:) = (target2train==iclass);
    end
    
    % draw process
    
    output = svm_output;
    target = svm_target;
    
    n_class_target = size(target,1);
    n_class_output = size(output,1);
    
    font_size = 10;
    n_bins = 20;
    
    bin_centers = linspace(-1,1,n_bins);
    
    for iclass_target = 1:n_class_target
        for iclass_output = 1:n_class_output
            subplot(n_class_target, n_class_output, (iclass_target-1)*n_class_output+iclass_output);
            [y,x]=hist(output(iclass_output,target(iclass_target,:)==1),bin_centers);
            plot(x,y/sum(y),'Color',dist_v_colors(ifolds,:),'LineWidth',2.0);
            
            title(sprintf('Class %i - Classifier %i',iclass_target,iclass_output),'FontSize',font_size,'FontWeight','bold');
            xlabel('Value','FontSize', font_size,'FontWeight', 'bold');
            ylabel('Prob.','FontSize', font_size,'FontWeight', 'bold');
            xlim([-1 1]);
            ylim([0 1]);
            set(gca,'FontWeight','bold');
            set(gca,'XTick',[-1 -.75 -.5 -.25 0.0 .25 .5 .75 1]);
            set(gca,'YTick',[0.0 0.2 0.4 0.6 0.8 1]);
            hold on;
            grid on;
        end
    end
    H{ifolds} = sprintf('Fold %i',ifolds);
end

figure(h);

legend(H,'Location','BestOutside');

set(gcf,'PaperUnits', 'normal','PaperPosition',[0 0 2 1]);
set(gcf,'Position',[.0 .0 .9 .9]);
print(sprintf('%s/pict/RunNNTrainCVAnalysis_output_dist_per_class_per_fold.png',outputpath),'-dpng','-r0')

%saveas(h,sprintf('%s/pict/RunNNTrainCVAnalysis_output_dist_per_class_per_fold.png',outputpath));
close(h);
clear('H');

ifolds = 2;
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

load(sprintf('%s/mat/RunSVMTrainCV_fold_%02i.mat',outputpath,ifolds));

svm_output = [];
svm_target = [];

for iclass = 1:numel(class_labels)
    svm_output(iclass,:) = svmclassify(svm_classifier{iclass},data_norm')';
    svm_target(iclass,:) = (target2train==iclass);
end

h = plot_confusion_matrix(target,output,0.9);
saveas(h,sprintf('%s/pict/RunSVMTrainCVAnalysis_output_confusion_fold.png',outputpath));
close(h);

% % show event in first and second PCA and errors

% % DO IT IN CORRECT WAY!!!!!!
%

%close all;

