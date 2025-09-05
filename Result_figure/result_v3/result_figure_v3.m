clear all;
close all;
clc

BL2 = zeros(6, 10, 99);
count = 0;
for v = 5 : 5 : 30
    count = count + 1;
    load(['test_WCL_v' num2str(v) '_a' num2str(v * 0.2) '.0_25dBm_ICC_ir_3CNN_1LSTM_160epoch_m=9_v2.mat']);
    BL2(v/5, :, :) = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));
end

BL3 = zeros(6, 10, 99);
count = 0;
for v = 5 : 5 : 30
    count = count + 1;
    load(['test_WCL_v' num2str(v) '_a' num2str(v * 0.2) '.0_25dBm_ICC_ir_3CNN_1LSTM_160epoch_m=4_v2_running1.mat']);
    BL3(v/5, :, :) = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));
end

BL4 = zeros(6, 10, 99);
count = 0;
for v = 5 : 5 : 30
    count = count + 1;
    load(['test_WCL_v' num2str(v) '_a' num2str(v * 0.2) '.0_25dBm_ODE_3CNN_1LSTM_160epoch_v1.mat']);
    BL4(count, :, :) = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));
end

BL1 = zeros(6, 10, 99);
count = 0;
for v = 5 : 5 : 30
    count = count + 1;
    load(['test_WCL_v' num2str(v) '_a' num2str(v * 0.2) '.0_25dBm_LSTM_ir_3CNN_1LSTM_v2.mat']);
    BL1(count, :, :) = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));
end

load(['EKF_result2.mat']);
BL0 = beam_loss_sery_total(2 : end, :, :);
BL0 = permute(BL0 ,[3, 2, 1]);
 

figure;
hold on;
grid on;
xlabel('Normalized prediction instant $\overline{\tau}$', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
plot(0.01 : 0.01 : 0.99, squeeze(mean(BL0(1, :, :), 2)), 'm-','linewidth', 1.5);
plot(0.01 : 0.01 : 0.99, squeeze(mean(BL1(1, :, :), 2)), '-', 'Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(0.01 : 0.01 : 0.99,squeeze(mean(BL2(1, :, :), 2)), '-','Color',[0.18    0.54   0.34], 'linewidth', 1.5);
plot(0.01 : 0.01 : 0.99, squeeze(mean(BL3(1, :, :), 2)), 'b-', 'linewidth', 1.5);
plot(0.01 : 0.01 : 0.99, squeeze(mean(BL4(1, :, :), 2)), 'r-', 'linewidth', 1.5);
legend( 'EKF [3] ','Conventional LSTM [9] ', 'Cascaded LSTM with $9$ predictions [10] ','Cascaded LSTM with $4$ predictions [10] ', 'Proposed ODE-LSTM  ', ...
    'interpreter', 'latex');

figure;
hold on;
grid on;
xlabel('Normalized prediction instant $\overline{\tau}$', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
plot(0.01 : 0.01 : 0.99, squeeze(mean(BL0(4, :, :), 2)),  'm-.','linewidth', 1.5);
plot(0.01 : 0.01 : 0.99, squeeze(mean(BL1(4, :, :), 2)), '-.', 'Color',[1    0.6   0.07],  'linewidth', 1.5);
plot(0.01 : 0.01 : 0.99, squeeze(mean(BL2(4, :, :), 2)), '-.', 'Color',[0.18    0.54   0.34], 'linewidth', 1.5);
plot(0.01 : 0.01 : 0.99, squeeze(mean(BL3(4, :, :), 2)), 'b-.', 'linewidth', 1.5);
plot(0.01 : 0.01 : 0.99, squeeze(mean(BL4(4, :, :), 2)), 'r-.', 'linewidth', 1.5);
 
legend( 'EKF [3] ','Conventional LSTM [9] ', 'Cascaded LSTM with $9$ predictions [10] ','Cascaded LSTM with $4$ predictions [10] ', 'Proposed ODE-LSTM  ', ...
    'interpreter', 'latex');
 

figure;
hold on;
grid on;
xlabel('Beam training instant (s)', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
ylim([0.7 1]);
set(gca,'yticklabel',[0.7 : 0.05 : 1]);
plot(0 : 0.16 : 1.44, squeeze(mean(BL0(1, :, :), 3)), 'm-*','linewidth', 1.5);
plot(0 : 0.16 : 1.44, squeeze(mean(BL1(1, :, :), 3)), '-v', 'Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(0 : 0.16 : 1.44, squeeze(mean(BL2(1, :, :), 3)), '-+','Color',[0.18    0.54   0.34], 'linewidth', 1.5);
plot(0 : 0.16 : 1.44, squeeze(mean(BL3(1, :, :), 3)), 'b-^', 'linewidth', 1.5);
plot(0 : 0.16 : 1.44, squeeze(mean(BL4(1, :, :), 3)), 'r-o', 'linewidth', 1.5);
legend( 'EKF [3] ','Conventional LSTM [9] ', 'Cascaded LSTM with $9$ predictions [10] ','Cascaded LSTM with $4$ predictions [10] ', 'Proposed ODE-LSTM  ', ...
    'interpreter', 'latex');

figure;
hold on;
grid on;
xlabel('Beam training instant (s)', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
plot(0 : 0.16 : 1.44, squeeze(mean(BL0(4, :, :), 3)),  'm-.*','linewidth', 1.5);
plot(0 : 0.16 : 1.44, squeeze(mean(BL1(4, :, :), 3)), '-.v', 'Color',[1    0.6   0.07],  'linewidth', 1.5);
plot(0 : 0.16 : 1.44, squeeze(mean(BL2(4, :, :), 3)), '-.+', 'Color',[0.18    0.54   0.34], 'linewidth', 1.5);
plot(0 : 0.16 : 1.44, squeeze(mean(BL3(4, :, :), 3)), 'b-.^', 'linewidth', 1.5);
plot(0 : 0.16 : 1.44, squeeze(mean(BL4(4, :, :), 3)), 'r-.o', 'linewidth', 1.5);
 
legend( 'EKF [3] ','Conventional LSTM [9] ', 'Cascaded LSTM with $9$ predictions [10] ','Cascaded LSTM with $4$ predictions [10] ', 'Proposed ODE-LSTM  ', ...
    'interpreter', 'latex');

figure;
hold on;
grid on;
xlabel('UE velocity $v$ (m/s)', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
plot(5 : 5 : 30, squeeze(mean(mean(BL0(:, :, :), 3), 2)),  'm-*', 'linewidth', 1.5);
plot(5 : 5 : 30, squeeze(mean(mean(BL1(:, :, :), 3), 2)), '-v', 'Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(5 : 5 : 30, squeeze(mean(mean(BL2(:, :, :), 3), 2)), '-+', 'Color',[0.18    0.54   0.34], 'linewidth', 1.5);
plot(5 : 5 : 30, squeeze(mean(mean(BL3(:, :, :), 3), 2)), 'b-^', 'linewidth', 1.5);
plot(5 : 5 : 30, squeeze(mean(mean(BL4(:, :, :), 3), 2)), 'r-o', 'linewidth', 1.5);
legend('EKF [3]','Conventional LSTM [9]','Cascaded LSTM with $9$ predictions [10]','Cascaded LSTM with $4$ predictions [10]', 'Proposed ODE-LSTM', ...
     'interpreter', 'latex');