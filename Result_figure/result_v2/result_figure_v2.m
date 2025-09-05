clear all;
close all;
clc;

load('WCL_v5_a1.0_25dBm_LSTM_ir_3CNN_1LSTM_v2.mat');
BL_mat1 = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));
BGR_m1 = mean(BL_mat1,1);
BGR_t1 = mean(BL_mat1,2);
 
load('WCL_v5_a1.0_25dBm_ICC_ir_3CNN_1LSTM_160epoch_m=4_v2.mat');
BL_mat2 = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));
BGR_m2 = mean(BL_mat2,1);
BGR_t2 = mean(BL_mat2,2);
 
load('WCL_v5_a1.0_25dBm_ODE_ir_3CNN_1LSTM_160epoch_v1.mat')
BL_mat3 = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));
BGR_m3 = mean(BL_mat3,1);
BGR_t3 = mean(BL_mat3,2);
 




load('WCL_v20_a4.0_25dBm_LSTM_ir_3CNN_1LSTM_v1.mat');
BL_mat4 = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));
BGR_m4 = mean(BL_mat4,1);
BGR_t4 = mean(BL_mat4,2);
 
load('WCL_v20_a4.0_25dBm_ICC_ir_3CNN_1LSTM_160epoch_m=4_v2.mat');
BL_mat5 = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));
BGR_m5 = mean(BL_mat5,1);
BGR_t5 = mean(BL_mat5,2);
 
load('WCL_v20_a4.0_25dBm_ODE_ir_3CNN_1LSTM_160epoch_v1.mat')
BL_mat6 = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));
BGR_m6 = mean(BL_mat6,1);
BGR_t6 = mean(BL_mat6,2);

figure;
hold on;
grid on;
xlabel('Beam training instant (s)', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
plot(BGR_t1, 'm-v',  'linewidth', 1.5 );
hold on;
plot(BGR_t2,'b-^',  'linewidth', 1.5 );
hold on;
plot(BGR_t3, 'r-o', 'linewidth', 1.5 );
hold on;
plot(BGR_t4, 'm-.v',  'linewidth', 1.5 );
hold on;
plot(BGR_t5,'b-.^',  'linewidth', 1.5 );
hold on;
plot(BGR_t6, 'r-.o', 'linewidth', 1.5 );

legend('Conventional LSTM [8] ($v=5$m/s)', 'Cascaded LSTM [9] ($v=5$m/s)', 'Proposed ODE-LSTM ($v=5$m/s)', ...
    'Conventional LSTM [8] ($v=20$m/s)', 'Cascaded LSTM [9] ($v=20$m/s)', 'Proposed ODE-LSTM ($v=20$m/s)', ...
    'interpreter', 'latex');



figure;
hold on;
grid on;
xlabel('Normalized prediction instant (s)', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
plot(BGR_m1, 'm-v',  'linewidth', 1.5 );
hold on;
plot(BGR_m2,'b-^',  'linewidth', 1.5 );
hold on;
plot(BGR_m3, 'r-o', 'linewidth', 1.5 );
hold on;
plot(BGR_m4, 'm-.v',  'linewidth', 1.5 );
hold on;
plot(BGR_m5,'b-.^',  'linewidth', 1.5 );
hold on;
plot(BGR_m6, 'r-.o', 'linewidth', 1.5 );

legend('Conventional LSTM [8] ($v=5$m/s)', 'Cascaded LSTM [9] ($v=5$m/s)', 'Proposed ODE-LSTM ($v=5$m/s)', ...
    'Conventional LSTM [8] ($v=20$m/s)', 'Cascaded LSTM [9] ($v=20$m/s)', 'Proposed ODE-LSTM ($v=20$m/s)', ...
    'interpreter', 'latex');
 
 
 