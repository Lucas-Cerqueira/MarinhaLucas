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

% function add
addpath(genpath('functions'));

% if LofarData has created...
if(exist(sprintf('%s/LofarData.mat',outputpath),'file'))
    answer = input(sprintf('File OUTPUTPATH/LofarData.mat exists, perform LofarAnalysis.m? [Y,n]'),'s');
    if strcmp(answer,'Y')
    else if strcmp(answer,'n')
            clear all;
            close all;
            clc;
            return;
        end
    end
end


if(~exist(sprintf('%s/RawData.mat',outputpath),'file'))
    fprintf('DO CREATERAWDATA\n');
    return;
else
    load(sprintf('%s/RawData.mat',outputpath));
end

% do LOFAR Analysis

n_pts_fft = 0;
n_pts_fft = input(sprintf('Number of FFT Points [default: 1024]: '));


if n_pts_fft == 0
    n_pts_fft = 1024;
end


decimation_rate = 0;
decimation_rate = input(sprintf('Decimation Ratio [default: 3]: '));

if decimation_rate == 0
    decimation_rate = 3;
end

spectrum_bins_left = 0;
spectrum_bins_left = input(sprintf('Spectrum Bins left for Analysis [default: 400]: '));

if spectrum_bins_left == 0
    spectrum_bins_left = 400;
end

show_plot = false;
answer = input(sprintf('Show Plots? [Y,n] '),'s');

if strcmp(answer,'Y')
    show_plot = true;
end

% LOFAR Parameters
num_overlap = 0;

norm_parameters.lat_window_size = 10;
norm_parameters.lat_gap_size = 1;
norm_parameters.threshold = 1.3;

fprintf('\nLOFAR Computing - All runs together\n');
data_lofar = [];
for iclass = 1:numel(class_labels) % All Classes
    fprintf('%s - All Runs in RawData file\n',class_labels{iclass});
    if decimation_rate >=1
        data_lofar.(class_labels{iclass}) = decimate(data.(class_labels{iclass}),decimation_rate,10,'FIR');
        Fs=fs/decimation_rate;
    else
        data_lofar.(class_labels{iclass}) = data.(class_labels{iclass});
        Fs=fs;
    end
    
    [intensity,f,t]=spectrogram(data_lofar.(class_labels{iclass}),hanning(n_pts_fft),num_overlap,n_pts_fft,Fs);
    intensity = abs(intensity);
    intensity=intensity./tpsw(intensity);
    intensity=log10(intensity);
    intensity(find(intensity<-.2))=0;
    data_lofar.(class_labels{iclass}) = intensity(1:spectrum_bins_left,:); % William
    
    h = [];
    if show_plot
        h = figure('visible','on');
        colormap jet;
    else
        h = figure('visible','off');
        colormap jet;
    end
    
    imagesc(f(1:spectrum_bins_left),t,data_lofar.(class_labels{iclass})');
    
    if decimation_rate >=1
        title(sprintf('LOFARgram for %s with Decimation Ratio: %d and %d FFT Points',class_labels{iclass},decimation_rate,n_pts_fft),'FontSize', 15,'FontWeight', 'bold');
    else
        title(sprintf('LOFARgram for %s with %d FFT Points',class_labels{iclass},n_pts_fft),'FontSize', 15,'FontWeight', 'bold');
    end
    
    ylabel('Time (seconds)','FontSize', 15,'FontWeight', 'bold');
    xlabel('Frequency (Hz)','FontSize', 15,'FontWeight', 'bold');
    colorbar;
    saveas(h,sprintf('%s/lofargram_%s.png',outputpath,class_labels{iclass}));
    close(h);
end


fprintf('\nLOFAR Computing - Each run\n');
rundata_lofar = [];
for iclass = 1:numel(class_labels) % All Classes
    for irun = 1:length(rundata.(class_labels{iclass}))
        fprintf('%s - %d Run in RawData file\n',class_labels{iclass}, irun);
        if decimation_rate >=1
            rundata_lofar.(class_labels{iclass}){irun} = decimate(rundata.(class_labels{iclass}){irun},decimation_rate,10,'FIR');
            Fs=fs/decimation_rate;
        else
            rundata_lofar.(class_labels{iclass}){irun} = rundata.(class_labels{iclass}){irun};
            Fs=fs;
        end
        
        [intensity,f,t]=spectrogram(rundata_lofar.(class_labels{iclass}){irun},hanning(n_pts_fft),num_overlap,n_pts_fft,Fs);
        intensity = abs(intensity);
        intensity=intensity./tpsw(intensity);
        intensity=log10(intensity);
        intensity(find(intensity<-.2))=0;
        rundata_lofar.(class_labels{iclass}){irun} = intensity(1:400,:); % William
        
        h = figure('visible','off');
        colormap jet;
        imagesc(f,t,intensity');
        
        if decimation_rate >=1
            title(sprintf('LOFARgram for %s (Run %d) with Decimation Ratio: %d and %d FFT Points',class_labels{iclass},irun,decimation_rate,n_pts_fft),'FontSize', 15,'FontWeight', 'bold');
        else
            title(sprintf('LOFARgram for %s (Run %d) with %d FFT Points',class_labels{iclass},irun,n_pts_fft),'FontSize', 15,'FontWeight', 'bold');
        end
        
        ylabel('Time (seconds)','FontSize', 15,'FontWeight', 'bold');
        xlabel('Frequency (Hz)','FontSize', 15,'FontWeight', 'bold');
        colorbar;
        saveas(h,sprintf('%s/lofargram_%s_run_%d.png',outputpath,class_labels{iclass},irun));
        close(h);
    end
end

fprintf('\nCreating LOFAR Data File\n');
save(sprintf('%s/LofarData.mat',outputpath),'decimation_rate','Fs','num_overlap','norm_parameters','data_lofar','rundata_lofar','n_pts_fft');

close all;

rmpath(genpath('functions'));
