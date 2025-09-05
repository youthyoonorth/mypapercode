%Ray-tracing scenario
params.scenario= 'O1_28';           % The adopted ray tracing scenarios [check the available scenarios at www.aalkhateeb.net/DeepMIMO.html]

%Dynamic Scenario Scenes
params.scene_first = 1;
params.scene_last = 1;

%%%% DeepMIMO parameters set %%%%
% Active base stations
params.active_BS = 1 : 12;          % Includes the numbers of the active BSs (values from 1-18 for 'O1')

% Active users
params.active_user_first = 1;       % The first row of the considered receivers section (check the scenario description for the receiver row map)
params.active_user_last = 2751;     % The last row of the considered receivers section (check the scenario description for the receiver row map)

% Subsampling of active users
% Setting both subsampling parameters to 1 activate all the users indicated previously
params.row_subsampling = 1;       % Randomly select round(row_subsampling*(active_user_last-params.active_user_first)) rows
params.user_subsampling = 1;      % Randomly select round(user_subsampling*number_of_users_in_row) users in each row

% Number of BS Antenna
params.num_ant_x=1;                 % Number of the UPA antenna array on the x-axis
params.num_ant_y=64;                % Number of the UPA antenna array on the y-axis
params.num_ant_z=1;                 % Number of the UPA antenna array on the z-axis
                                    % Note: The axes of the antennas match the axes of the ray-tracing scenario

params.num_ant_MS_x=1;              % Number of the UPA antenna array on the x-axis
params.num_ant_MS_y=1;              % Number of the UPA antenna array on the y-axis
params.num_ant_MS_z=1;              % Number of the UPA antenna array on the z-axis

% Antenna spacing
params.ant_spacing=.5;              % ratio of the wavelength; for half wavelength enter .5
params.ant_spacing_MS=.5;           % ratio of the wavelength; for half wavelength enter .5

% System parameters
params.enable_BS2BSchannels=0;      % Enable (1) or disable (0) generation of the channels between basestations
params.bandwidth=0.1;               % The bandwidth in GHz
params.transmit_power = 20;         % The transmit power in dBm
params.pulse_shaping = 1;           % Pulse shaping choices: (1) No pulse shaping and matched filter,
                                    % (2) sinc pulse shaping and matched filter, (3) raised cosine pulse shaping and matched filter,
                                    % (4) user-defined pulse shaping and matched filter (Please edit the code in .\DeepMIMO_functions\userdefined_pulse.m).
params.activate_RX_ideal_LPF = 0;   % Activate receive ideal LPF for pulse shaping choices 2, 3, and 4.
params.rolloff_factor = 0.2;        % Raised cosine rolloff factor (a value between 0 and 1)
params.pulse_upsampling_factor = 100;% Upsampling factor for generating sinc or raised cosine pulses

% Channel parameters
params.activate_FD_channels = 0;    % 1: activate frequency domain (FD) channel generation for OFDM systems
                                    % 0: activate instead time domain (TD) channel impulse response generation for non-OFDM systems
params.num_paths=25;                % Maximum number of paths to be considered (a value between 1 and 25), e.g., choose 1 if you are only interested in the strongest path

% if params.activate_FD_channels == 1
% OFDM parameters
params.num_OFDM=256;                % Number of OFDM subcarriers
params.OFDM_sampling_factor=1;      % The constructed channels will be calculated only at the sampled subcarriers (to reduce the size of the dataset)
params.OFDM_limit=64;               % Only the first params.OFDM_limit subcarriers will be considered when constructing the channels
params.cyclic_prefix_ratio=1.0;     % Cyclic prefix ratio from the OFDM symbol length. The OFDM symbol length = params.num_OFDM.

% default data save mode
params.saveDataset = 0;