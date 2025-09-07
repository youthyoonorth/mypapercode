%Ray-tracing scenario
params.scenario= 'O1_28';           % The adopted ray tracing scenarios [check the available scenarios at www.aalkhateeb.net/DeepMIMO.html]

%Dynamic Scenario Scenes
params.scene_first = 1;
params.scene_last = 1;

%%%% DeepMIMO parameters set %%%%
% Active base stations
% Consider several distributed BSs for large-scale MIMO deployment
params.active_BS = [1 5 9];          % Distributed active BS indices (values from 1-18 for 'O1')

% Active users
params.active_user_first = 1;       % The first row of the considered receivers section (check the scenario description for the receiver row map)
params.active_user_last = 2751;     % The last row of the considered receivers section (check the scenario description for the receiver row map)

% Subsampling of active users
% Setting both subsampling parameters to 1 activate all the users indicated previously
params.row_subsampling = 1;       % Randomly select round(row_subsampling*(active_user_last-params.active_user_first)) rows
params.user_subsampling = 1;      % Randomly select round(user_subsampling*number_of_users_in_row) users in each row

% Number of antennas at the BS and UE (x, y, z)
params.bs_antenna.shape = [8 64 8];    % Enhanced array to support 3D beamforming
params.ue_antenna.shape = [1 1 2];     % UE antenna configuration for 3D reception

% Antenna spacing
params.bs_antenna.spacing = .5;        % ratio of the wavelength; for half wavelength enter .5
params.ue_antenna.spacing = .5;        % ratio of the wavelength; for half wavelength enter .5

% System parameters
params.enable_BS2BS = 0;            % Enable (1) or disable (0) generation of the channels between basestations
params.bandwidth = 0.1;             % The bandwidth in GHz

% Channel parameters
params.OFDM_channels = 0;          % 1: activate frequency domain (FD) channel generation for OFDM systems
                                    % 0: activate instead time domain (TD) channel impulse response generation for non-OFDM systems
params.num_paths = 25;              % Maximum number of paths to be considered (1-25)

% OFDM parameters
params.num_OFDM = 256;              % Number of OFDM subcarriers
params.OFDM_sampling_factor = 1;    % The constructed channels will be calculated only at the sampled subcarriers
params.OFDM_limit = 64;             % Only the first params.OFDM_limit subcarriers will be considered when constructing the channels

% default data save mode
params.saveDataset = 0;
