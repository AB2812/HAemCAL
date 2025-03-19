clc; clear; close all;

%% **Step 1: Select Data Files**
wall_files = uipickfiles('Prompt', 'Select Wall Pressure Files');
p1_files = uipickfiles('Prompt', 'Select P1 Pressure Files');

if isempty(wall_files) || isempty(p1_files)
    error('No files selected.');
end

num_wall_files = length(wall_files);
num_p1_files = length(p1_files);

fprintf('\nTotal Wall Files Selected: %d\n', num_wall_files);
fprintf('Total P1 Files Selected: %d\n', num_p1_files);

if num_wall_files ~= num_p1_files
    error('Mismatch in number of wall and P1 files.');
end

%% **Step 2: Extract Node Coordinates**
wall_data = readmatrix(wall_files{1});
node_numbers = wall_data(:,1);  % Node number (for reference)
wall_coords = wall_data(:, 2:4);  % Extract (X, Y, Z) coordinates
[num_nodes, ~] = size(wall_coords);

fprintf('\nNumber of Nodes in Wall Geometry: %d\n', num_nodes);

%% **Step 3: Initialize Pressure Variables**
wall_pressure_sum = zeros(num_nodes, 1);
p1_pressure_global_sum = 0;
total_p1_nodes = 0;

%% **Step 4: Compute Global Time & Node-Averaged P1 Pressure**
for t = 1:num_p1_files
    p1_data = readmatrix(p1_files{t});
    fprintf('Importing P1 File: %s\n', p1_files{t});

    % Sum P1 pressure values across all nodes and time steps
    p1_pressure_global_sum = p1_pressure_global_sum + sum(p1_data(:, end));
    total_p1_nodes = total_p1_nodes + size(p1_data, 1);
end

% Compute final global time-averaged P1 value
p1_pressure_global_avg = p1_pressure_global_sum / total_p1_nodes;
fprintf('Global Time & Node-Averaged P1 Pressure: %.4f Pa\n', p1_pressure_global_avg);

%% **Step 5: Compute Time-Averaged Wall Pressure at Each Node**
for t = 1:num_wall_files
    wall_data = readmatrix(wall_files{t});
    fprintf('Importing Wall File: %s\n', wall_files{t});

    if size(wall_data, 1) ~= num_nodes
        error('Mismatch in node count between wall data and expected nodes.');
    end

    wall_pressure = wall_data(:, end);
    wall_pressure_sum = wall_pressure_sum + wall_pressure;

    fprintf('Processing Time Step %d/%d\n', t, num_wall_files);
end

% Compute time-averaged wall pressure per node
wall_pressure_avg = wall_pressure_sum / num_wall_files;

%% **Step 6: Compute FFR for Each Node**
FFR = wall_pressure_avg / p1_pressure_global_avg;

%% **Step 7: Normalize FFR Values Using remap.m**
FFR_mapped = remap(FFR, [min(FFR) max(FFR)], [0 1]);

%% **Step 8: Correct 3D Point Cloud Visualization**
figure;
scatter3(wall_coords(:,1), wall_coords(:,2), wall_coords(:,3), 50, FFR_mapped, 'filled');

colormap(jet);
colorbar;
caxis([0 1]);  % Normalize color scale
title('FFR Distribution on Artery Wall (Correct 3D Geometry)');
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
axis equal;
grid on;

%% **Step 9: Export Results (Optional)**
output_data = [node_numbers, wall_coords, FFR_mapped];
writematrix(output_data, 'FFR_3D_Distribution.csv');

disp('FFR Calculation and 3D Visualization Completed.');
