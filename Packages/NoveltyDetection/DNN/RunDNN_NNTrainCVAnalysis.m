% Projeto Marinha do Brasil

% Autor: Natanael Junior (natmourajr@gmail.com)
% Laboratorio de Processamento de Sinais - UFRJ


% Objetivos: Analisar (com Validacao Cruzada) redes neurais para deteccao de novidade em sinais proveniente de navios

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

% create data struct adn analysis


load(sprintf('%s/TrainInformationDNN_NN_novelty_%s.mat',outputpath,class_labels{1}));
% choose best NN (best SP)
mat_best_init = zeros(numel(class_labels),length(possible_topo),n_folds);

% plot eff per threshold

n_thr = 50;
possible_thr = linspace(-1.1,1.1,n_thr);

v_eff = zeros(numel(class_labels),length(possible_topo),n_thr,n_folds,numel(class_labels));
v_sp = zeros(numel(class_labels),length(possible_topo),n_thr,n_folds);

for novelty_class = 1:numel(class_labels) % todas as classes
    
    fprintf('Novelty Class: %s \n',class_labels{novelty_class});
    
    if(~exist(sprintf('%s/TrainInformationDNN_NN_novelty_%s.mat',outputpath,class_labels{novelty_class}),'file')),
        continue;
    end
    
    load(sprintf('%s/TrainInformationDNN_NN_novelty_%s.mat',outputpath,class_labels{novelty_class}));
    
    data_novelty = data_lofar.(class_labels{novelty_class});
    
    for topo = 1:length(possible_topo)
        fprintf('Topo: %i\n', possible_topo(topo));
        for ifolds = 1:n_folds
            trained_nn = [];
            train_description = [];
            
            fprintf('Fold: %i of %i\n',ifolds, n_folds);
            if (develop_mode) && (ifolds > 2), break; end
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
                data_novelty_norm = mapstd('apply', data_novelty ,norm_fact);
            else
                % Min = -1 and Max = 1
                [~, norm_fact] = mapminmax(data2train(:,itrn));
                data_norm = mapminmax('apply', data2train ,norm_fact);
                data_novelty_norm = mapminmax('apply', data_novelty ,norm_fact);
            end
            
            load(sprintf('%s/mat/RunDNN_NNTrainCV_novelty_%s_topo_%02i_fold_%02i.mat',outputpath,class_labels{novelty_class},possible_topo(topo),ifolds));
            
            data_norm = result';
            data_novelty_norm = novelty_result';
            
            %Analysing NN
            sp = -99;
            
            for i_init = 1:n_init
                if (develop_mode) && (i_init > 2), break; end
                fprintf('Topo: %i - iFold: %i of %i - Init: %i of %i\n',possible_topo(topo), ifolds, n_folds, i_init, n_init);
                output = sim(trained_nn{i_init},data_norm);
                output_novelty = sim(trained_nn{i_init},data_novelty_norm);
                target = target2train_norm;
                [SP,Eff] = computeSP(target,output);
                if SP > sp
                    sp = SP;
                    mat_best_init(novelty_class,topo,ifolds) = i_init;
                    
                    % threshold analysis
                    eff = [];
                    for i_thr = 1:size(possible_thr,2)
                        for iclass = 1:numel(class_labels)-1 % todas as classes
                            [a_test,id] = max(output,[],1);
                            v_eff(novelty_class,topo,i_thr,ifolds,iclass) = sum(a_test >= possible_thr(i_thr) & target(1,:)== 1 & id==iclass)/sum(target(1,:)== 1);
                        end
                        v_eff(novelty_class,topo,i_thr,ifolds,numel(class_labels)) = sum(  max(output_novelty,[],1) < possible_thr(i_thr) )/length(output_novelty);
                    end
                end
            end
        end
    end
end


% Draw plots
v_colors = [0 0 1; 1 0 0; 0 0.5 0; 0.75 0.75 0];

for topo = 1:length(possible_topo)
    for novelty_class = 1:numel(class_labels)
        h = figure;
        for iclass = 1:numel(class_labels)
            if iclass == novelty_class, continue; end
            h_errorbar = [];
            if iclass < novelty_class
                H{iclass} = class_labels{iclass};
                mean_eff = mean(v_eff(novelty_class,topo,:,:,iclass),4);
                var_eff = var(v_eff(novelty_class,topo,:,:,iclass),[],4);
                h_errorbar = errorbar(possible_thr, mean_eff, var_eff,'Color',v_colors(iclass,:));
            else % if iclass < novelty_class
                H{iclass-1} = class_labels{iclass};
                mean_eff = mean(v_eff(novelty_class,topo,:,:,iclass-1),4);
                var_eff = var(v_eff(novelty_class,topo,:,:,iclass-1),[],4);
                h_errorbar = errorbar(possible_thr, mean_eff, var_eff,'Color',v_colors(iclass,:));
            end % if iclass < novelty_class
            set(h_errorbar,'Linewidth',2.0);
            hold on;
        end % for iclass
        h_errorbar = errorbar(possible_thr, mean(v_eff(novelty_class,topo,:,:,numel(class_labels)),4),var(v_eff(novelty_class,topo,:,:,numel(class_labels)),[],4),'Color',[0 0 0]);
        set(h_errorbar,'Linewidth',2.0);
        H{numel(class_labels)} = 'Novelty Class';

        legend(H, 'Location', 'SouthWest');

        title(sprintf('NN (Topo %i): Class Efficiency per Threshold (Novelty: %s)',possible_topo(topo),class_labels{novelty_class}),'FontSize', 15,'FontWeight', 'bold');
        xlabel(sprintf('Threshold'),'FontSize', 15,'FontWeight', 'bold');
        ylabel(sprintf('Class Efficiency'),'FontSize', 15,'FontWeight', 'bold');

        xlim([-1 1]);
        ylim([0 1]);

        hold off;
        grid on;

        set(gca,'fontweight','bold');

        % Inset Plot
%         handler = axes('Position',[.2 .3 .5 .5]);
%         box off;
% 
%         for iclass = 1:numel(class_labels) % todas as classes
%             if iclass == novelty_class, continue; end
% 
%             h_errorbar = [];
%             if iclass < novelty_class
%                 mean_eff = mean(v_eff(novelty_class,topo,:,:,iclass),4);
%                 var_eff = var(v_eff(novelty_class,topo,:,:,iclass),[],4);
%                 h_errorbar = errorbar(possible_thr(end/2:end),mean_eff(end/2:end),var_eff(end/2:end),'Color',v_colors(iclass,:));
%             else
%                 mean_eff = mean(v_eff(novelty_class,topo,:,:,iclass-1),4);
%                 var_eff = var(v_eff(novelty_class,topo,:,:,iclass-1),[],4);
%                 h_errorbar = errorbar(possible_thr(end/2:end), mean_eff(end/2:end) ,var_eff(end/2:end),'Color',v_colors(iclass,:));
%             end
%             set(h_errorbar,'Linewidth',2.0)
%             hold on;
%         end
%         mean_eff = mean(v_eff(novelty_class,topo,:,:,numel(class_labels)),4);
%         var_eff = var(v_eff(novelty_class,topo,:,:,numel(class_labels)),[],4);
% 
%         h_errorbar = errorbar(possible_thr(end/2:end), mean_eff(end/2:end), var_eff(end/2:end),'Color',[0 0 0]);
%         set(h_errorbar,'Linewidth',2.0);
% 
%         grid on;
%         xlim([0.5 1]);
%         ylim([0 1]);
%         set(gca,'XTick',0.5:0.05:1);
%         set(gca,'YTick',[0:0.1:0.3 0.8:0.1:1]);
%         set(gca,'fontweight','bold');

        saveas(h,sprintf('%s/pict/RunDNN_NN_Representation_TrainCVAnalysis_AfterTraining_topo_%i_novelty_%s_class_eff_per_thr.png',outputpath,possible_topo(topo),class_labels{novelty_class}));
        %close(h);
        clear('H');
    end %for novelty_class
end % for topo


% plot NN output dist per class per novelty train
choose_topo = 1;
choose_fold = 1;
choose_thr = 0.0;

for novelty_class = 1:numel(class_labels)
    h = figure;
    load(sprintf('%s/mat/RunDNN_NNTrainCV_novelty_%s_topo_%02i_fold_%02i.mat',outputpath,class_labels{novelty_class},possible_topo(choose_topo),choose_fold));
    load(sprintf('%s/TrainInformationDNN_NN_novelty_%s.mat',outputpath,class_labels{novelty_class}));
    
    data_novelty = data_lofar.(class_labels{novelty_class});
    
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
        data_novelty_norm = mapstd('apply', data_novelty ,norm_fact);
    else
        % Min = -1 and Max = 1
        [~, norm_fact] = mapminmax(data2train(:,itrn));
        data_norm = mapminmax('apply', data2train ,norm_fact);
        data_novelty_norm = mapminmax('apply', data_novelty ,norm_fact);
    end
    
    data_norm = result';
    data_novelty_norm = data_norm (:,target2train==novelty_class);
    
%     h = figure;
%     for i=1:size(target2train, 1)
%         for class = 1:numel(class_labels)
%             subplot (size(target2train, 1),numel(class_labels), numel(class_labels)*(i-1) + class);
%             hold on;
%             
%             if class == novelty_class
%                 [y,x] = hist (sim(trained_nn{mat_best_init(choose_fold)},data_novelty_norm), 50);
%                 plot (x, y, 'k-', 'linewidth', 2.0);
%             else
%                 [y,x] = hist (sim(trained_nn{mat_best_init(choose_fold)},data_norm (:,target2train==class)), 50);
%                 plot (x, y, '-', 'color', v_colors (class,:), 'linewidth', 2.0);
%             end
%             %xlim([0 1]);
%         end
%         hold off;
%     end

    
    old_target_plus_novelty = [target2train_norm; zeros(1,length(target2train_norm))]; % add a new dimension
    buffer = zeros(numel(class_labels),length(data_novelty_norm));
    buffer(numel(class_labels),:) = 1;
    
    target_plus_novelty = [old_target_plus_novelty buffer];
    %target_plus_novelty = target2train_norm;
    output_plus_novelty = [sim(trained_nn{mat_best_init(choose_fold)},data_norm) sim(trained_nn{mat_best_init(choose_fold)},data_novelty_norm)];
    
    dim_target = size(target_plus_novelty,1);
%     dim_output = size(output_plus_novelty,1);
    dim_output = numel (class_labels) - 1;
    
    font_size = 10;
    n_bins = 50;
    
    bin_centers = linspace(-1,1,n_bins);
    for iclass_target = 1:dim_target
        for iclass_output = 1:dim_output
            subplot(dim_target, dim_output, (iclass_target-1)*dim_output+iclass_output);
            
            [y,x]=hist(output_plus_novelty(iclass_output,target(1,:)==1),bin_centers);
            
            if (iclass_target >= novelty_class) && iclass_target <= numel(class_labels)-1
                plot(x,y/sum(y),'Color',v_colors(iclass_target+1,:),'LineWidth',2.0);
                title(sprintf('%s - Neuron %i',class_labels{iclass_target+1},iclass_output),'FontSize',font_size,'FontWeight','bold');
            else
                if iclass_target <= numel(class_labels)-1
                    plot(x,y/sum(y),'Color',v_colors(iclass_target,:),'LineWidth',2.0);
                    title(sprintf('%s - Neuron %i',class_labels{iclass_target},iclass_output),'FontSize',font_size,'FontWeight','bold');
                else
                    % legend
                    hold on;
                    for leg_class = 1:numel(class_labels)
                        if leg_class == novelty_class, continue; end
                        plot(0.0,0.0,'Color',v_colors(leg_class,:),'LineWidth',2.0);
                        if leg_class < novelty_class
                            H{leg_class} = class_labels{leg_class};
                        else
                            H{leg_class-1} = class_labels{leg_class};
                        end
                    end
                    
                    plot(x,y/sum(y),'Color',[0 0 0],'LineWidth',2.0);
                    title(sprintf('Novelty Class - Neuron %i',iclass_output),'FontSize',font_size,'FontWeight','bold');
                    hold off;
                    H{numel(class_labels)} = 'Novelty Class';
                end
                
            end
            
            xlabel('Value','FontSize', font_size,'FontWeight', 'bold');
            ylabel('Prob.','FontSize', font_size,'FontWeight', 'bold');
            
%             xlim([-1 1]);
            ylim([0 1]);
            set(gca,'FontWeight','bold');
%             set(gca,'XTick',[-1 -.75 -.5 -.25 0.0 .25 .5 .75 1]);
%             set(gca,'YTick',[0.0 0.2 0.4 0.6 0.8 1]);
            grid on;
            % legend : TO DO!!!
        end
    end
    
    warning off;
    %legend(H,'Location','SouthEastOutside');
    warning on;
    
%     set(gcf,'PaperUnits', 'normal','PaperPosition',[0 0 2 1]);
%     set(gcf,'Position',[.0 .0 .9 .9]);
    
    print(h,sprintf('%s/pict/RunDNN_NN_Representation_AfterTraining_TrainCVAnalysis_topo_%i_novelty_%s_ultGrafico.png',outputpath,possible_topo(choose_topo),class_labels{novelty_class}),'-dpng','-r0');
%    close(h);
    
end % for novelty_class

