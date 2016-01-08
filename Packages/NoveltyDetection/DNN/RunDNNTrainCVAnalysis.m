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

data2train = [];
target2train = [];
target2train_norm = [];

% create data struct
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


% plot eff per threshold

n_thr = 50;
possible_thr = linspace(-0.1,1.1,n_thr);

v_eff = zeros(numel(class_labels),n_thr,n_folds,numel(class_labels));
v_sp = zeros(numel(class_labels),n_thr,n_folds);

for novelty_class = 1:numel(class_labels) % todas as classes
    load(sprintf('%s/TrainInformationDNN_novelty_%s.mat',outputpath,class_labels{novelty_class}));
    
    fprintf('Novelty Class: %s \n',class_labels{novelty_class});
    
    if(~exist(sprintf('%s/TrainInformationDNN_novelty_%s.mat',outputpath,class_labels{novelty_class}),'file')),
        continue;
    end
    
    
    data_novelty = data_lofar.(class_labels{novelty_class});
    output = [];
    output_novelty = [];
    
    for indexTopo=1:size(topos,1)
        fprintf ('Topo: %i of %i\n', indexTopo, size(topos,1));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for ifolds = 1%:n_folds
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
        
            load(sprintf('%s/mat/RunDNNTrainCV_DNN-Representation_novelty_%s_fold_%i_topo_%i.mat',outputpath,class_labels{novelty_class},ifolds,indexTopo));

            output = (sim_nn(dnn_representation,data_norm(:,target2train~=novelty_class)'))';
            output_novelty = (sim_nn(dnn_representation,data_novelty'))';
            
            if (classificationTraining)
                %Analysing NN
                sp = -99;

                if (develop_mode) && (i_init > 2), break; end


                fprintf('indexTopo: %i of %i\n', indexTopo, size (topos,1));
                fprintf('iFold: %i of %i\n', ifolds, n_folds);
                
                target = target2train_norm;
                [SP,Eff] = computeSP(target,output);
                if SP > sp
                    sp = SP;

                    % threshold analysis
                    eff = [];
                    for i_thr = 1:size(possible_thr,2)
                        for iclass = 1:numel(class_labels)-1 % todas as classes
                            [a_test,id] = max(output,[],1);
                            v_eff(novelty_class,i_thr,ifolds,iclass,indexTopo) = sum(a_test >= possible_thr(i_thr) & target(iclass,:)== 1 & id==iclass)/sum(target(iclass,:)== 1);
                        end
                        v_eff(novelty_class,i_thr,ifolds,numel(class_labels),indexTopo) = sum(  max(output_novelty,[],1) < possible_thr(i_thr) )/length(output_novelty);
                    end
                end
            end
        end
    end
end


% Draw plots
v_colors = [0 0 1; 1 0 0; 0 0.5 0; 0.75 0.75 0];

for novelty_class = 1:numel(class_labels)
    load(sprintf('%s/TrainInformationDNN_novelty_%s.mat',outputpath,class_labels{novelty_class}));
    data_novelty = data_lofar.(class_labels{novelty_class});

    H = {};
    
    for indexTopo = 1:size(topos,1)
  
        if (classificationTraining)
            h = figure;
            for iclass = 1:numel(class_labels)
                if iclass == novelty_class, continue; end
                h_errorbar = [];
                if iclass < novelty_class
                    H{iclass} = class_labels{iclass};
                   % mean_eff = mean(v_eff(novelty_class,:,:,iclass),4);
                    mean_eff = mean(v_eff(novelty_class,:,:,iclass,indexTopo),3);            
                   % var_eff = var(v_eff(novelty_class,:,:,iclass),[],4);
                    var_eff = var(v_eff(novelty_class,:,:,iclass,indexTopo),[],3);            
                    h_errorbar = errorbar(possible_thr,mean_eff,var_eff,'Color',v_colors(iclass,:));
                else % if iclass < novelty_class
                    H{iclass-1} = class_labels{iclass};
                    %mean_eff = mean(v_eff(novelty_class,:,:,iclass-1),4);
                    mean_eff = mean(v_eff(novelty_class,:,:,iclass-1,indexTopo),3);            
                    %var_eff = var(v_eff(novelty_class,:,:,iclass-1),[],4);
                    var_eff = var(v_eff(novelty_class,:,:,iclass-1,indexTopo),[],3);            
                    h_errorbar = errorbar(possible_thr,mean_eff,var_eff,'Color',v_colors(iclass,:));

                end % if iclass < novelty_class
                set(h_errorbar,'Linewidth',2.0);
                hold on;
            end % for iclass

            %mean_eff = mean(v_eff(novelty_class,:,:,numel(class_labels)),4);        
            mean_eff = mean(v_eff(novelty_class,:,:,numel(class_labels),indexTopo),3);            
            %var_eff = var(v_eff(novelty_class,:,:,numel(class_labels)),[],4);
            var_eff = var(v_eff(novelty_class,:,:,numel(class_labels),indexTopo),[],3);
            h_errorbar = errorbar(possible_thr, mean_eff,var_eff,'Color',[0 0 0]);
            set(h_errorbar,'Linewidth',2.0);
            H{numel(class_labels)} = 'Novelty Class';

            legend(H, 'Location', 'SouthWest');

            title(sprintf('DNN Class Efficiency per Threshold (Novelty: %s)(Topo: %i)',class_labels{novelty_class},indexTopo),'FontSize', 15,'FontWeight', 'bold');
            xlabel(sprintf('Threshold'),'FontSize', 15,'FontWeight', 'bold');
            ylabel(sprintf('Class Efficiency'),'FontSize', 15,'FontWeight', 'bold');

            xlim([0 1]);
            ylim([0 1]);

            hold off;
            grid on;

            set(gca,'fontweight','bold');

            % Inset Plot
        %    handler = axes('Position',[.2 .3 .5 .5]);
        %    box off;


        % 
        %     for iclass = 1:numel(class_labels) % todas as classes
        %         if iclass == novelty_class, continue; end
        % 
        %         h_errorbar = [];
        %         if iclass < novelty_class
        %             mean_eff = mean(v_eff(novelty_class,:,:,iclass),4);
        %             mean_eff = mean(v_eff(novelty_class,:,:,iclass),3);            
        %             var_eff = var(v_eff(novelty_class,:,:,iclass),[],4);
        %             var_eff = var(v_eff(novelty_class,:,:,iclass),[],3);
        %             h_errorbar = errorbar(possible_thr(end/2:end),mean_eff(end/2:end),var_eff(end/2:end),'Color',v_colors(iclass,:));
        %         else
        %             mean_eff = mean(v_eff(novelty_class,:,:,iclass-1),4);
        %             mean_eff = mean(v_eff(novelty_class,:,:,iclass-1),3);            
        %             var_eff = var(v_eff(novelty_class,:,:,iclass-1),[],4);
        %             var_eff = var(v_eff(novelty_class,:,:,iclass-1),[],3);
        %             h_errorbar = errorbar(possible_thr(end/2:end), mean_eff(end/2:end) ,var_eff(end/2:end),'Color',v_colors(iclass,:));
        %         end
        %         set(h_errorbar,'Linewidth',2.0)
        %         hold on;
        %     end
        %     mean_eff = mean(v_eff(novelty_class,:,:,numel(class_labels)),4);
        %     var_eff = var(v_eff(novelty_class,:,:,numel(class_labels)),[],4);
        % 
        %     h_errorbar = errorbar(possible_thr(end/2:end), mean_eff(end/2:end), var_eff(end/2:end),'Color',[0 0 0]);
        %     set(h_errorbar,'Linewidth',2.0);
        % 
        %     grid on;
        %     xlim([0.5 1]);
        %     ylim([0 1]);
        %     set(gca,'XTick',0.5:0.05:1);
        %     set(gca,'YTick',[0:0.1:0.3 0.8:0.1:1]);
        %     set(gca,'fontweight','bold');

            saveas(h,sprintf('%s/pict/RunDNNTrainCVAnalysis_novelty_%s_topo_%i_class_eff_per_thr.png',outputpath,class_labels{novelty_class},indexTopo));
        %    close(h);
            clear('H');
            %%%%%%%%%%%%%%%%%%%%%%%%%
            for ifolds=1:n_folds
                load(sprintf('%s/mat/RunDNNTrainCV_DNN-Representation_novelty_%s_fold_%i_topo_%i.mat',outputpath,class_labels{novelty_class},ifolds,indexTopo));
                output = sim_nn(dnn_representation,data2train(:,target2train~=novelty_class)');
                output_novelty = sim_nn(dnn_representation,data_novelty');
            
                h = figure;
                suptitle(sprintf('Classification Output (Novelty: %s)(Topo: %i)(Fold: %i)',class_labels{novelty_class},indexTopo,ifolds));
                hold on;
                for classIndex=1:numel(class_labels)
                    for neuron=1:3
                        subplot (numel(class_labels), 3, 3*(classIndex-1) + neuron)
                        if classIndex == novelty_class
                            [y,x] = hist (output_novelty(:, neuron)', 50);
                            plot (x, y, 'k-', 'linewidth', 2.0);
                        else
                            [y,x] = hist (output (target2train==classIndex, neuron)', 50);
                            plot (x, y, '-', 'color', v_colors (classIndex,:), 'linewidth', 2.0);
                        end
                    xlim([0 1]);    
                    end
                end
                
            end
            
        else
            %%%%%%%%%%%%%%%%%%%%%%%%
            for ifolds=1%:n_folds
                load(sprintf('%s/mat/RunDNNTrainCV_DNN-Representation_novelty_%s_fold_%i_topo_%i.mat',outputpath,class_labels{novelty_class},ifolds,indexTopo));
                output = sim_nn(dnn_representation,data2train(:,target2train~=novelty_class)');
                output_novelty = sim_nn(dnn_representation,data_novelty');
                fprintf ('Novelty: %i\n', novelty_class);
    
                h = figure;
                %%%%%
                hold on;
                %title(sprintf('SAE Last Layer Representation (Novelty: %s)(Topo: %i)',class_labels{novelty_class},indexTopo),'FontSize', 15,'FontWeight', 'bold');
                suptitle(sprintf('SAE Last Layer Representation (Novelty: %s)(Topo: %i)(Fold: %i)',class_labels{novelty_class},indexTopo,ifolds));
                hold off;
                %%%%%%
                
                
                switch topos(indexTopo,end)
                    case 4
                    for i=1:4
                        subplot (2,2,i);
                        hold on;
                         %%%%%%%%%
                        xlabel(sprintf('Output'),'FontSize', 8,'FontWeight', 'bold');
                        ylabel(sprintf('Propability'),'FontSize', 8,'FontWeight', 'bold');
                        %%%%%%%%%
                        for class = 1:numel(class_labels)                          
                          if class == novelty_class
                              [y,x] = hist (output_novelty(:,i)', 50);
                              %plot (x, y/sum(y*diff(x(1:2))), 'k-', 'linewidth', 2.0);
                              plot (x, y, 'k-', 'linewidth', 2.0);
                          else
                              H{class} = class_labels{class};
                              
                              [y,x] = hist (output (target2train==class,i)', 50);
                              %plot (x, y/sum(y*diff(x(1:2))), '-', 'color', v_colors (class,:), 'linewidth', 2.0);
                              plot (x, y, '-', 'color', v_colors (class,:), 'linewidth', 2.0);
                          end
                          %xlim([0 1]);
                          %ylim([0 1]);
                        end
                        hold off;
                    end
                    
                    otherwise
                    for i=1:10
                        subplot (2,5,i);
                        hold on;
                         %%%%%%%%%
                        xlabel(sprintf('Output'),'FontSize', 8,'FontWeight', 'bold');
                        ylabel(sprintf('Propability'),'FontSize', 8,'FontWeight', 'bold');
                        %%%%%%%%%
                        for class = 1:numel(class_labels)
                          if class == novelty_class
                              [y,x] = hist (output_novelty(:,i)', 50);
                              %plot (x, y/sum(y*diff(x(1:2))), 'k-', 'linewidth', 2.0);
                              plot (x, y, 'k-', 'linewidth', 2.0);
                          else
                              H{class} = class_labels{class};
                              
                              [y,x] = hist (output (target2train==class,i)', 50);
                              %plot (x, y/sum(y*diff(x(1:2))), '-', 'color', v_colors (class,:), 'linewidth', 2.0);
                              plot (x, y, '-', 'color', v_colors (class,:), 'linewidth', 2.0);
                          end
                          %xlim([0 1]);
                          %ylim([0 1]);
                        end
                        hold off;
                    end
                end   
                
                H{novelty_class} = 'Novelty Class';
                legend(H, 'Location', 'SouthWest');                
                
                saveas(h,sprintf ('%s/pict/representation_output_novelty_%s_topo_%i_fold_%i.png', outputpath, class_labels{novelty_class},indexTopo,ifolds));                
                close (h);
                clear ('H');
                
            end
        end
    end
end %for novelty_class