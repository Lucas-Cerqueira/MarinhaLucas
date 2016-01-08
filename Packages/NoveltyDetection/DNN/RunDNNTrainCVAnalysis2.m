% Projeto Marinha do Brasil

% Autor: Natanael Junior (natmourajr@gmail.com)
% Laboratorio de Processamento de Sinais - UFRJ


% Objetivos: Treinar (com Validacao Cruzada) redes neurais para classificacao de navios

fprintf('Starting %s.m\n',mfilename('fullpath'));

clear all;
close all;

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
end    

SP = [];
Eff = [];

for novelty_class = 1:numel(class_labels) % todas as classes
    load(sprintf('%s/TrainInformationDNN_novelty_%s.mat',outputpath,class_labels{novelty_class}));
    
    fprintf('Novelty Class: %s \n',class_labels{novelty_class});
    
    if(~exist(sprintf('%s/TrainInformationDNN_novelty_%s.mat',outputpath,class_labels{novelty_class}),'file')),
        continue;
    end
     
    data_novelty = data_lofar.(class_labels{novelty_class});
   
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
        dnn_output = [];
        
        load(sprintf('%s/mat/RunDNNTrainCV_AfterTraining_NN-Representation_novelty_%s.mat',outputpath,class_labels{novelty_class}));
        fprintf('iFold: %i of %i\n', ifolds, n_folds);
        dnn_output = (sim_nn(nn_representation,data_norm(:,target2train~=novelty_class)'))';
        [SP{ifolds},Eff{ifolds}] = computeSP(target2train_norm,dnn_output);
    end
end

% Analysis

%SP Analysis

mat_sp = zeros(n_folds);
mat_eff = zeros(n_folds,numel(class_labels));
mat_init = zeros(n_folds);

for ifolds = 1:n_folds
    if (develop_mode) && (ifolds > 2), continue; end;
    sp_aux = -99;
    
    if SP{ifolds} > sp_aux
        sp_aux = SP{ifolds};
        mat_sp(ifolds) = SP{ifolds};
        eff_aux = Eff{ifolds};
        for iclass = 1:numel(class_labels)
            mat_eff(ifolds,iclass) = eff_aux(iclass);
        end

    end
end


% compute sp mean and var
mean_sp = mean(mat_sp,1);
var_sp = var(mat_sp,[],1);

h = figure;
h_errorbar = errorbar(possible_topo,mean_sp,var_sp,'Color',[0 0 0]);
set(h_errorbar,'Linewidth',2.0);
H = 'SP Index'; legend(H, 'Location', 'SouthEast');

title(sprintf('NN: SP Index per TOPO'),'FontSize', 15,'FontWeight', 'bold');
xlabel(sprintf('TOPO'),'FontSize', 15,'FontWeight', 'bold');
ylabel(sprintf('SP Index'),'FontSize', 15,'FontWeight', 'bold');
ylim([0 1]);

grid on;

set(gca,'XTick',possible_topo);

saveas(h,sprintf('%s/pict/RunNNTrainCVAnalysis_sp_per_topo.png',outputpath));
close(h);
clear('H');

%show output dist per neuron

choose_topo = 2;

dist_v_colors = jet(n_folds);

h = figure;

for ifolds = 1:n_folds
    if (develop_mode) && (ifolds > 2), continue; end;
    
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
    
    
    load(sprintf('%s/mat/RunNNTrainCV_topo_%02i_fold_%02i.mat',outputpath,choose_topo,ifolds))
    
    target = target2train_norm;
    
    % choosing init: max(SP)
    output = sim(trained_nn{mat_init(choose_topo,ifolds)},data_norm);
    
    % draw process
    
    n_class_target = size(target,1);
    n_class_output = size(output,1);
    
    font_size = 10;
    n_bins = 50;
    
    bin_centers = linspace(-1,1,n_bins);
    
    for iclass_target = 1:n_class_target
        for iclass_output = 1:n_class_output
            subplot(n_class_target, n_class_output, (iclass_target-1)*n_class_output+iclass_output);
            [y,x]=hist(output(iclass_output,target(iclass_target,:)==1),bin_centers);
            plot(x,y/sum(y),'Color',dist_v_colors(ifolds,:),'LineWidth',2.0);
            
            title(sprintf('%s - Neuron %i',class_labels{iclass_target},iclass_output),'FontSize',font_size,'FontWeight','bold');
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
print(sprintf('%s/pict/RunNNTrainCVAnalysis_output_dist_per_neuron_per_fold.png',outputpath),'-dpng','-r0')

%saveas(h,sprintf('%s/pict/RunNNTrainCVAnalysis_output_dist_per_neuron_per_fold.png',outputpath));
close(h);
clear('H');

% compute Eff mean and var
mean_eff = mean(mat_eff,2);
var_eff = var(mat_eff,[],2);

h = figure;

v_colors = [0 0 1; 1 0 0; 0 0.5 0; 0.75 0.75 0];

for iclass = 1:numel(class_labels)
    h_errorbar = errorbar(possible_topo,mean_eff(:,iclass),var_eff(:,iclass),'Color',v_colors(iclass,:));
    set(h_errorbar,'Linewidth',2.0);
    hold on;
    H{iclass} = class_labels{iclass};
end

title(sprintf('NN: Class Efficiency per TOPO'),'FontSize', 15,'FontWeight', 'bold');
xlabel(sprintf('TOPO'),'FontSize', 15,'FontWeight', 'bold');
ylabel(sprintf('Class Efficiency'),'FontSize', 15,'FontWeight', 'bold');
ylim([0 1]);
legend(H, 'Location', 'SouthEast');

grid on;
set(gca,'XTick',possible_topo);

saveas(h,sprintf('%s/pict/RunNNTrainCVAnalysis_eff_per_topo.png',outputpath));
close(h);
clear('H');

% show output errors in one fold

choose_fold = 1;
choose_topo = 2;

trn_id =  CVO.training(choose_fold);
tst_id =  CVO.test(choose_fold);

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


load(sprintf('%s/mat/RunNNTrainCV_topo_%02i_fold_%02i.mat',outputpath,choose_topo,choose_fold))

v_colors = [0 0 1; 1 0 0; 0 0.5 0; 0.75 0.75 0];

target = target2train_norm;

% choosing init: max(SP)
output = sim(trained_nn{mat_init(choose_topo,choose_fold)},data_norm);

warning off;
[~,~,indexes,~] = confusion(target,output);
h = plot_confusion_matrix(target,output,0.0);
saveas(h,sprintf('%s/pict/RunNNTrainCVAnalysis_output_confusion_per_neuron_per_fold.png',outputpath));
close(h);
warning on;

% draw process
n_class_target = size(target,1);
n_class_output = size(output,1);

font_size = 10;
n_bins = 50;

bin_centers = linspace(-1,1,n_bins);

for iclass_target = 1:n_class_target
    for iclass_output = 1:n_class_output
        subplot(n_class_target, n_class_output, (iclass_target-1)*n_class_output+iclass_output);
        
        
        for other_class = 1:n_class_output
            if iclass_target ~= other_class
                [y,x]=hist(output(iclass_output,indexes{iclass_target,other_class}),bin_centers);
                plot(x,y/sum(y),'Color',v_colors(other_class,:),'LineWidth',2.0);
            else
                plot(0.0, 0.0, 'Color',v_colors(other_class,:),'LineWidth',2.0);
            end
            hold on;
        end
        
        %         for iclass_error = 1:n_class_output
        %             if ~isempty(indexes{iclass_target,iclass_output}) && (iclass_target ~= iclass_error)
        %                 [y,x]=hist(output(iclass_output,indexes{iclass_target,iclass_error}),bin_centers);
        %                 plot(x,y/sum(y),'Color',v_colors(iclass_error,:),'LineWidth',2.0);
        %             else
        %                 plot(0.0, 0.0, 'Color',v_colors(iclass_error,:),'LineWidth',2.0)
        %             end
        %             hold on;
        %         end
        %
        title(sprintf('%s - Neuron %i',class_labels{iclass_target},iclass_output),'FontSize',font_size,'FontWeight','bold');
        xlabel('Value','FontSize', font_size,'FontWeight', 'bold');
        ylabel('Prob.','FontSize', font_size,'FontWeight', 'bold');
        xlim([-1 1]);
        ylim([0 1]);
        set(gca,'FontWeight','bold');
        set(gca,'XTick',[-1 -.75 -.5 -.25 0.0 .25 .5 .75 1]);
        set(gca,'YTick',[0.0 0.2 0.4 0.6 0.8 1]);
        
        grid on;
    end
end

warning off;
legend(class_labels,'Location','BestOutside');
warning on;

set(gcf,'PaperUnits', 'normal','PaperPosition',[0 0 2 1]);
set(gcf,'Position',[.0 .0 .9 .9]);
print(sprintf('%s/pict/RunNNTrainCVAnalysis_output_error_dist_per_neuron_per_fold.png',outputpath),'-dpng','-r0')

return;

close all;
clear all;
clc;
