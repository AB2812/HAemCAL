% FFR Calculation and Visualization

clc; clear; close all;

% Request simulation parameters from user
prompt = {'Simulation Start Time (s):', 'Simulation End Time (s):', 'Time Step Size (s):', 'Cardiac Cycle Length (s):'};
dlgtitle = 'Simulation Parameters';
dims = [1 50];
definput = {'0', '1', '0.01', '1'};
answer = inputdlg(prompt, dlgtitle, dims, definput);

if isempty(answer)
    error('No simulation parameters provided.');
end

sim_start = str2double(answer{1});
sim_end = str2double(answer{2});
time_step = str2double(answer{3});
cycle_length = str2double(answer{4});

% Calculate expected number of time steps
expected_time_steps = round((sim_end - sim_start) / time_step);

% Load P1 files
try
    p1_files = uipickfiles('Prompt', 'Select P1 (upstream) files');
    if isempty(p1_files)
        error('No P1 files selected.');
    end
catch
    error('Error selecting P1 files.');
end

num_files_p1 = length(p1_files);

if num_files_p1 ~= expected_time_steps
    error('Number of P1 files (%d) does not match expected time steps (%d).', num_files_p1, expected_time_steps);
end

% Load P2 files
try
    p2_files = uipickfiles('Prompt', 'Select P2 (downstream) files');
    if isempty(p2_files)
        error('No P2 files selected.');
    end
catch
    error('Error selecting P2 files.');
end

num_files_p2 = length(p2_files);

if num_files_p2 ~= expected_time_steps
    error('Number of P2 files (%d) does not match expected time steps (%d).', num_files_p2, expected_time_steps);
end

% Preallocate variables
ffr_time = zeros(expected_time_steps, 1); % FFR values over time
flow_time = linspace(sim_start, sim_end, expected_time_steps)'; % Flow time vector

% Loop through all files to calculate FFR
for t = 1:expected_time_steps
    % Load data for current time step
    p1_data = readmatrix(p1_files{t});
    p2_data = readmatrix(p2_files{t});

    % Extract static pressure column
    p1_pressure = p1_data(:, end); % Assuming last column contains pressure
    p2_pressure = p2_data(:, end);

    % Compute nodal average pressures
    p1_avg = mean(p1_pressure);
    p2_avg = mean(p2_pressure);

    % Compute FFR for current time step
    ffr = p2_avg / p1_avg;

    % Cap FFR to a maximum of 1.0
    ffr_time(t) = min(ffr, 1.0);
end

% Compute overall statistics
ffr_avg = mean(ffr_time);
[max_ffr, max_index] = max(ffr_time);
max_ffr_time = flow_time(max_index);

% Save results to CSV
output_data = [flow_time, ffr_time];
csv_filename = 'FFR_Time_Variation.csv';
writematrix(output_data, csv_filename);
fprintf('FFR results saved to: %s\n', csv_filename);

% Display statistics
fprintf('Average FFR over all time steps: %.4f\n', ffr_avg);
fprintf('Maximum FFR: %.4f at Flow Time: %.4f s\n', max_ffr, max_ffr_time);

% Plot FFR vs. Flow Time
figure;
plot(flow_time, ffr_time, '-o', 'LineWidth', 2);
xlabel('Flow Time (s)');
ylabel('FFR');
title('FFR vs. Flow Time');
grid on;
