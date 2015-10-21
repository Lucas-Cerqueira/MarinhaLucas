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
outputpath = sprintf('%s/Classification/NeuralNetwork',datapath);

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

% Training History text file
trainHistoryPath = sprintf ('%s/trainHistory.txt', outputpath);
if exist (trainHistoryPath, 'file') == 2
    historyFile = fopen (trainHistoryPath, 'r');
else
    error('DO %s/Packages/Classification/NeuralNetwork/RunNNTrainCV.m\n',workspace);
end

line = 0;
options = cell (0, 1);
while line ~= -1
    line = fgetl (historyFile);
    if line ~= -1
        trainInfo = strsplit (line, '#');
        load(sprintf('%s/%s/TrainInformationNN.mat',outputpath, trainInfo {1}));
        options {end+1, 1} = sprintf ('%s | Folds:%s | Init:%s | Develop:%s | Epochs:%i', trainInfo{1}, trainInfo{2}, trainInfo{3}, trainInfo{4}, n_epochs);
    end
end
fclose (historyFile);

options = cellstr (options);

if usejava ('desktop')
    [optionIndexVector, ok] = listdlg ('ListString', options, 'ListSize', [500 300]);
else
    arrayLength = size (options);
    arrayLength = arrayLength (1);
    fprintf ('Enter the number corresponding to the training you want to analyse\n');
    for i = 1:arrayLength
        fprintf ('%i - %s\n', i, options {i});
    end
    optionIndex = input ('');
end

for i=1:length(optionIndexVector)
    optionIndex = optionIndexVector (i)
    optionIndex = options {optionIndex}
    optionIndex = strsplit (optionIndex, ' | ')
    trainFolder = optionIndex {1}
    disp (trainFolder)
    % Train Information
    if(~exist(sprintf('%s/%s/TrainInformationNN.mat',outputpath, trainFolder),'file'))
        error('DO %s/DataHandler/RunNNTrainCVAnalysis\n',workspace);
    else
        load(sprintf('%s/%s/TrainInformationNN.mat',outputpath, trainFolder));
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

    SP = [];
    Eff = [];

    for topo = 1:length(possible_topo)
        fprintf('Topo: %i - Analysing NN Train\n', possible_topo(topo));
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
            nn_output = [];

            for i_init = 1:n_init
                if (develop_mode) && (i_init > 2), continue; end
                % ******
                %load(sprintf('%s/mat/RunNNTrainCV_topo_%02i_fold_%02i.mat',outputpath,possible_topo(topo),ifolds))
                load(sprintf('%s/%s/mat/RunNNTrainCV_topo_%02i_fold_%02i.mat',outputpath,trainFolder,possible_topo(topo),ifolds))
                fprintf('Topo: %i - iFold: %i of %i - Init: %i of %i\n',possible_topo(topo), ifolds, n_folds, i_init, n_init);
                nn_output{i_init} = sim(trained_nn{i_init},data_norm);
                [SP{topo,ifolds,i_init},Eff{topo,ifolds,i_init}] = computeSP(target2train_norm,nn_output{i_init});
            end
        end
    end

    % Analysis

    %SP Analysis

    mat_sp = zeros(length(possible_topo),n_folds);
    mat_eff = zeros(length(possible_topo),n_folds,numel(class_labels));
    mat_init = zeros(length(possible_topo),n_folds);

    for topo = 1:length(possible_topo)
        for ifolds = 1:n_folds
            if (develop_mode) && (ifolds > 2), continue; end;
            sp_aux = -99;
            for i_init = 1:n_init
                if (develop_mode) && (i_init > 2), continue; end;
                if SP{topo,ifolds,i_init} > sp_aux
                    sp_aux = SP{topo,ifolds,i_init};
                    mat_sp(topo,ifolds) = SP{topo,ifolds,i_init};
                    mat_init(topo,ifolds) = i_init;
                    eff_aux = Eff{topo,ifolds,i_init};
                    for iclass = 1:numel(class_labels)
                        mat_eff(topo,ifolds,iclass) = eff_aux(iclass);
                    end

                end
            end
        end
    end


    % compute sp mean and var
    mean_sp = mean(mat_sp,2);
    var_sp = var(mat_sp,[],2);

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

    saveas(h,sprintf('%s/%s/pict/RunNNTrainCVAnalysis_sp_per_topo.png',outputpath, trainFolder));
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

        % *********
        load(sprintf('%s/%s/mat/RunNNTrainCV_topo_%02i_fold_%02i.mat',outputpath,trainFolder,choose_topo,ifolds))

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
    print(sprintf('%s/%s/pict/RunNNTrainCVAnalysis_output_dist_per_neuron_per_fold.png',outputpath, trainFolder),'-dpng','-r0')

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

    saveas(h,sprintf('%s/%s/pict/RunNNTrainCVAnalysis_eff_per_topo.png',outputpath, trainFolder));
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


    load(sprintf('%s/%s/mat/RunNNTrainCV_topo_%02i_fold_%02i.mat',outputpath,trainFolder,choose_topo,choose_fold))

    v_colors = [0 0 1; 1 0 0; 0 0.5 0; 0.75 0.75 0];

    target = target2train_norm;

    % choosing init: max(SP)
    output = sim(trained_nn{mat_init(choose_topo,choose_fold)},data_norm);

    warning off;
    [~,~,indexes,~] = confusion(target,output);
    h = plot_confusion_matrix(target,output,0.0);
    saveas(h,sprintf('%s/%s/pict/RunNNTrainCVAnalysis_output_confusion_per_neuron_per_fold.png',outputpath, trainFolder));
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
    print(sprintf('%s/%s/pict/RunNNTrainCVAnalysis_output_error_dist_per_neuron_per_fold.png',outputpath, trainFolder),'-dpng','-r0');
end

close all;
clear all;
clc;
