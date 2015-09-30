% Projeto Marinha do Brasil

%Autor: Natanael Junior (natmourajr@gmail.com)
% Laboratorio de Processamento de Sinais - UFRJ

% iniciando o script
clear all;
close all;
clc;

fprintf('Starting %s.m\n',mfilename('fullpath'));


% System var. point to external folders
inputpath = getenv('INPUTDATAPATH');
outputpath = getenv('OUTPUTDATAPATH');

% if raw_data has created...
if(exist(sprintf('%s/RawData.mat',outputpath),'file'))
    answer = input(sprintf('File OUTPUTPATH/RawData.mat exists, perform CreateRawData.m? [Y,n]'),'s');
    if strcmp(answer,'Y')
    else if strcmp(answer,'n')
            clear all;
            close all;
            clc;
            return;
        end
    end
end


fprintf('\n\nDevelop Mode: First Run of all Class\n');
fprintf('Full Test Mode: All Run of all Class\n\n');

answer = input(sprintf('Do create_raw_data.m in develop mode?\n[Y,n]'),'s');

develop = false;

if strcmp(answer,'Y')
	develop = true;
else
	develop = false;
end


class_labels{1} = 'ClassA';
class_labels{2} = 'ClassB';
class_labels{3} = 'ClassC';
class_labels{4} = 'ClassD';


v_runs = [5 10 10 10];

% Data per Run
rundata = [];
for iclass = 1:numel(class_labels) % All classes
	aux = [];
	for irun = 0:v_runs(iclass)-1 % All runs if develop == false
		if develop && irun > 1 
			continue;
		end

		if iclass == 3 && irun == 6 
			continue;
		end

		fprintf('Reading %s Run %i\n',class_labels{iclass}, irun+1);
		%fprintf('read: %s\n',sprintf('%s/SONAR/Classification/%s/navio%i%i.wav',inputpath,class_labels{iclass},iclass,irun))
        
		[aux{irun+1},fs] = wavread(sprintf('%s/SONAR/Classification/%s/navio%i%i.wav',inputpath,class_labels{iclass},iclass,irun));
	end
	% one class with only 9 runs
	if ~develop && iclass == 3 
		aux(7) = [];
	end
	rundata.(class_labels{iclass}) = aux;
end

% Full Data
data = [];
for iclass = 1:numel(class_labels) % All classes
	aux = [];
	for irun = 1:length(rundata.(class_labels{iclass}))
		aux = [aux ; rundata.(class_labels{iclass}){irun}];
	end
	data.(class_labels{iclass}) = aux;
end

save(sprintf('%s/RawData.mat',outputpath),'fs','data','rundata','class_labels');

% all machine learning proccess with same number of data
n_folds = 10;
n_init = 10;
develop_mode = develop;

save(sprintf('%s/TrainInformation.mat',outputpath),'develop_mode','n_folds','n_init');

clear all;
close all;
