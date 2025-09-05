clear all;
close all;
clc

BL2 = zeros(6, 10, 9);
count = 0;
for v = 5 : 5 : 30
    count = count + 1;
    load(['WCL_v' num2str(v) '_a' num2str(v * 0.2) '.0_25dBm_ICC_3CNN_1LSTM_160epoch_m=4_v2.mat']);
    BL2(count, :, :) = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));
end

BL3 = zeros(6, 10, 9);
count = 0;
for v = 5 : 5 : 30
    count = count + 1;
    load(['WCL_v' num2str(v) '_a' num2str(v * 0.2) '.0_25dBm_ODE_3CNN_1LSTM_160epoch_v1.mat']);
    BL3(count, :, :) = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));
end

BL1 = zeros(6, 10, 9);
count = 0;
for v = 5 : 5 : 30
    count = count + 1;
    load(['WCL_v' num2str(v) '_a' num2str(v * 0.2) '.0_25dBm_LSTM_3CNN_1LSTM_160epoch_v1.mat']);
    BL1(count, :, :) = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));
end

BL0 = zeros(6, 10, 9);
load(['EKF_result.mat']);
BL0 = beam_loss_sery_total(2 : end, :, :);
BL0 = permute(BL0 ,[3, 2, 1]);

figure;
hold on;
grid on;
xlabel('Normalized prediction instant $\overline{\tau}$', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL0(1, :, :), 2)), '-*','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL1(1, :, :), 2)), 'm-v', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL2(1, :, :), 2)), 'b-^', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL3(1, :, :), 2)), 'r-o', 'linewidth', 1.5);

plot(0.1 : 0.1 : 0.9, squeeze(mean(BL0(4, :, :), 2)),  '-.*','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL1(4, :, :), 2)), 'm-.v',  'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL2(4, :, :), 2)), 'b-.^', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL3(4, :, :), 2)), 'r-.o', 'linewidth', 1.5);
legend('EKF [3] ($v=5$m/s)','Conventional LSTM [8] ($v=5$m/s)', 'Cascaded LSTM [9] ($v=5$m/s)', 'Proposed ODE-LSTM ($v=5$m/s)', ...
    'EKF [3] ($v=20$m/s)','Conventional LSTM [8] ($v=20$m/s)', 'Cascaded LSTM [9] ($v=20$m/s)', 'Proposed ODE-LSTM ($v=20$m/s)', ...
    'interpreter', 'latex');

figure;
hold on;
grid on;
xlabel('Beam training instant (s)', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
xticks([0 : 0.25 : 1.5]);
set(gca,'xticklabel',[0 : 0.25 : 1.5]);
plot(0 : 0.16 : 1.44, squeeze(mean(BL0(1, :, :), 3)),  '-*','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(0 : 0.16 : 1.44, squeeze(mean(BL1(1, :, :), 3)), 'm-v',  'linewidth', 1.5);
plot(0 : 0.16 : 1.44, squeeze(mean(BL2(1, :, :), 3)), 'b-^', 'linewidth', 1.5);
plot(0 : 0.16 : 1.44, squeeze(mean(BL3(1, :, :), 3)), 'r-o', 'linewidth', 1.5);

plot(0 : 0.16 : 1.44, squeeze(mean(BL0(4, :, :), 3)), '-.*','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(0 : 0.16 : 1.44, squeeze(mean(BL1(4, :, :), 3)), 'm-.v',  'linewidth', 1.5);
plot(0 : 0.16 : 1.44, squeeze(mean(BL2(4, :, :), 3)), 'b-.^', 'linewidth', 1.5);
plot(0 : 0.16 : 1.44, squeeze(mean(BL3(4, :, :), 3)), 'r-.o', 'linewidth', 1.5);
legend('EKF [3] ($v=5$m/s)','Conventional LSTM [8] ($v=5$m/s)', 'Cascaded LSTM [9] ($v=5$m/s)', 'Proposed ODE-LSTM ($v=5$m/s)', ...
    'EKF [3] ($v=20$m/s)','Conventional LSTM [8] ($v=20$m/s)', 'Cascaded LSTM [9] ($v=20$m/s)', 'Proposed ODE-LSTM ($v=20$m/s)', ...
    'interpreter', 'latex');

figure;
hold on;
grid on;
xlabel('UE velocity $v$ (m/s)', 'interpreter', 'latex');
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex');
plot(5 : 5 : 30, squeeze(mean(mean(BL0(:, :, :), 3), 2)),  '-*','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(5 : 5 : 30, squeeze(mean(mean(BL1(:, :, :), 3), 2)), 'm-v',  'linewidth', 1.5);
plot(5 : 5 : 30, squeeze(mean(mean(BL2(:, :, :), 3), 2)), 'b-^', 'linewidth', 1.5);
plot(5 : 5 : 30, squeeze(mean(mean(BL3(:, :, :), 3), 2)), 'r-o', 'linewidth', 1.5);
legend('EKF [3]','Conventional LSTM [8]', 'Cascaded LSTM [9]', 'Proposed ODE-LSTM', ...
    'interpreter', 'latex');