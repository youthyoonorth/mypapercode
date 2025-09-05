clear all;
close all;
clc

addpath('DeepMIMO_functions')
% total row number of UE distribution
row_num = 2751;
% UE number in each row
UE_row_num = 181;
% mmWave BS number
BS_num = 12;
% candidate mmWave channel
MM_channel = zeros(BS_num, UE_row_num, 64);

% mmWave channel parameters
params = read_params('parameters_mm.m');

% for each row
for i = 1 : row_num
    % select active users (this row)
    params.active_user_first = i;
    params.active_user_last = i;
    print_info = ['mmWave dataset ' num2str(i) 'th row generation started']
    % generate corresponding channels and parameters
    [dataset_MM, params_MM] = DeepMIMO_generator(params);
    % save channels into matrices
    for j = 1 : BS_num
        for k = 1 : UE_row_num
            MM_channel(j, k, :) = squeeze(sum(dataset_MM{j}.user{k}.channel, 3));
        end
    end
    % save channels into files
    fprintf('\n Saving the DeepMIMO Dataset ...')
    sfile_DeepMIMO = ['./MM_dataset/MM_DeepMIMO_dataset_' num2str(i) '_row.mat'];
    save(sfile_DeepMIMO,'MM_channel', '-v7.3');
end
fprintf('\n DeepMIMO Dataset Generation completed \n')
