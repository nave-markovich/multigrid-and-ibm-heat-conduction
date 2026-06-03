function NumericsQ7()
% =========================================================================
% Project: Advanced Numerical Methods - Question 7
% Task: Solve the unsteady 2D heat equation with an insulated immersed
%       circular hole.
% Method: Explicit direct-forcing Immersed Boundary Method (IBM)
%         using probe/ghost sampling and explicit Eulerian time integration.
% =========================================================================
clc; close all; tic;

%% 1. Parameters Definition
L  = 1.0;                  % Domain length
R  = 0.2;                  % Hole radius
N  = 200;                  % Structured grid size
h  = L / N;                % Uniform cell size (counts as dx = dy)

k1 = 1e-3;                 % Conductivity in most of the plate
k2 = 100.0;                % Conductivity in top-right quadrant

photo = 2;                 % Snapshot flag for t = 1 and t = 5

k_max = max(k1, k2);                     % Maximum conductivity for CFL estimate
dt_max_stable = h^2 / (4 * k_max);       % Explicit stability estimate
safety_factor = 0.9;                     % Safety factor for stable integration
dt = dt_max_stable * safety_factor;      % Final explicit time step

t_final = 500.0;           % Maximum simulation time limit

% Extremely small dt leads to minimal temperature changes per step,
% which may trigger false convergence before physical equilibrium.
tolerance = dt * 1e-6;     % Adaptive steady-state tolerance

% Adjustable intervals. Increasing these values reduces computational
% overhead and speeds up the simulation.
log_interval  = 1000;      % Progress print interval
plot_interval = 10000;     % Monitoring plot interval

%% 2. Initialization and Assembly (Using Local Functions)
% Setup Eulerian grid and conductivity field
[x_vec, xc, yc, K] = Setup_Grid_And_Conductivity(N, h, k1, k2);

% External Dirichlet boundary values: T = 1 - 4y^2
T_boundary_values = (1 - 4 * (x_vec.^2)).';

% Build Lagrangian boundary and IBM operators
[eps_L, eta_L, M_probe, M_ghost, M_spread_ghost, K_lag,total_dist] = ...
    Assemble_IBM_Operators(xc, yc, K, h, R);

% Assemble explicit Eulerian operator and BC contribution
[L_mat, b_bc] = AssembleLaplacian(K, h, T_boundary_values, N);

fprintf('Total setup time before time integration: %.2f sec\n', toc);

%% 3. Initial Visualization
figure(1); colormap('jet');
Monitor_Simulation(zeros(N, N), zeros(N, N), 0, 0, ...
                   x_vec, x_vec, eps_L, eta_L, ...
                   tolerance, log_interval, plot_interval);

%% 4. Unsteady Time Integration Loop
fprintf('Starting time integration for Question 7...\n');

T = zeros(N, N);           % Initial temperature field
n = 0;
current_t = 0;

T_steady = [];
t_steady = [];

while current_t < t_final
    n = n + 1;
    current_t = n * dt;
    T_old_step = T;

    % Step 1: Explicit Eulerian predictor
    T_vec = T(:);
    T_star_vec = T_vec + dt * (L_mat * T_vec + b_bc);
    T_star = reshape(T_star_vec, N, N);

    % Step 2: Interpolate predictor field to probe and ghost rings
    T_p = M_probe * T_star_vec;
    T_g = M_ghost * T_star_vec;

    % Step 3: Compute Lagrangian forcing for zero-flux enforcement
    % This formula calculates the exact flux needed to make T_g = T_p
    % in a single time step, effectively blocking the flux.
    Q_lag = -(T_p - T_g) / total_dist;
    Q_lag = Q_lag / h;         % Dimensional scaling to match source term units
    Q_weighted = Q_lag .* K_lag; % Units adjustment

    % Step 4: Spread the forcing back to the Eulerian grid
    q_spread = M_spread_ghost * Q_weighted;
    Q_ibm = reshape(full(q_spread), N, N);

    % Step 5: Correct the Eulerian field
    % Apply the IBM force as a source term to the intermediate field
    T_new = T_star - dt * Q_ibm;

    % Step 6: Re-enforce external boundary conditions
    T_new(:, 1)   = T_boundary_values; 
    T_new(:, end) = T_boundary_values; 
    T_new(1, :)   = T_new(2, :);       
    T_new(end, :) = T_new(end-1, :);

    % Step 7: Save requested snapshots
    if photo == 2 && current_t >= 1.0
        T_snap1 = T_new;
        t_curr1 = current_t;
        photo = 3;
    end

    if photo == 3 && current_t >= 5.0
        T_snap2 = T_new;
        t_curr2 = current_t;
        photo = 0;
    end

    % Step 8: Monitor convergence and evolution
    if mod(n, log_interval) == 0
        [stop_sim, ~] = Monitor_Simulation(T_new, T_old_step, n, current_t, ...
                                           x_vec, x_vec, eps_L, eta_L, ...
                                           tolerance, log_interval, plot_interval);

        if stop_sim
            T = T_new;
            T_steady = T;
            t_steady = current_t;
            break;
        end
    end

    T = T_new;
end

if isempty(T_steady)
    T_steady = T;
    t_steady = current_t;
end

%% 5. Requested Output Plots
Plot_Snapshot(2, x_vec, T_snap1, t_curr1, eps_L, eta_L, ...
              'Snapshot at Time');
Plot_Snapshot(3, x_vec, T_snap2, t_curr2, eps_L, eta_L, ...
              'Snapshot at Time');
Plot_Snapshot(4, x_vec, T_steady, t_steady, eps_L, eta_L, ...
              'Steady-State Snapshot at Time');

fprintf('Simulation complete. Total CPU time: %.2f sec\n', toc);
end

% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function [x_vec, xc, yc, K] = Setup_Grid_And_Conductivity(N, h, k1, k2)
    % Initialize 1D grid coordinates
    x_vec = linspace(-0.5 + h/2, 0.5 - h/2, N);

    % Create 2D coordinate grids
    [xc, yc] = meshgrid(x_vec, x_vec);

    % Initialize conductivity field
    K = k1 * ones(N, N);

    % Assign k2 to the top-right quadrant
    K(xc >= 0 & yc >= 0) = k2;
end

function [eps_L, eta_L, M_probe, M_ghost, M_spread_ghost, K_lag,total_dist] = ...
    Assemble_IBM_Operators(xc, yc, K, h, R)
    fprintf('Assembling IBM operators...\n');

    l_factor = 2.0;            % Base divisor for the sampling offset
    l_offset = h / l_factor;   % Normal offset from the immersed boundary
    total_dist = 2 * l_offset;

    % Lagrangian points on the immersed boundary
    NL = ceil((2 * pi * R) / h);
    theta_L = linspace(0, 2*pi, NL+1).';
    theta_L(end) = [];

    eps_L = R * cos(theta_L);
    eta_L = R * sin(theta_L);

    % Exterior probe ring: Sensors located in the active fluid/solid domain
    eps_p = (R + l_offset) * cos(theta_L);
    eta_p = (R + l_offset) * sin(theta_L);
    M_probe = IBMatrix(xc, yc, eps_p, eta_p, h, N_from_grid(xc));

    % Interior ghost ring: Points inside the hole for flux-blocking force application
    eps_g = (R - l_offset) * cos(theta_L);
    eta_g = (R - l_offset) * sin(theta_L);
    M_ghost = IBMatrix(xc, yc, eps_g, eta_g, h, N_from_grid(xc));

    % Spreading operator for the ghost ring
    M_spread_ghost = M_ghost.';

    % Conductivity interpolated to the probe ring
    K_lag = M_probe * K(:);
end

function [L, b_bc] = AssembleLaplacian(K, h, T_boundary, N)
    fprintf('Assembling explicit Eulerian operator...\n');

    N2 = N * N;
    I = [];    J = [];    V = [];
    b_bc = zeros(N2, 1);

    % Column-major indexing: i = row (y), j = column (x)
    k_idx = @(i,j) i + (j-1) * N;

    for j = 1:N
        for i = 1:N
            row = k_idx(i,j);
            V_diag = 0;

            % West / left boundary: x = -0.5
            if j == 1
                coeff = (2 * K(i,j)) / (h^2);
                V_diag = V_diag - coeff;
                b_bc(row) = b_bc(row) + coeff * T_boundary(i);
            else
                kw = 2 * K(i,j) * K(i,j-1) / (K(i,j) + K(i,j-1));
                coeff = kw / (h^2);
                V_diag = V_diag - coeff;
                I = [I; row]; J = [J; k_idx(i,j-1)]; V = [V; coeff];
            end

            % East / right boundary: x = +0.5
            if j == N
                coeff = (2 * K(i,j)) / (h^2);
                V_diag = V_diag - coeff;
                b_bc(row) = b_bc(row) + coeff * T_boundary(i);
            else
                ke = 2 * K(i,j) * K(i,j+1) / (K(i,j) + K(i,j+1));
                coeff = ke / (h^2);
                V_diag = V_diag - coeff;
                I = [I; row]; J = [J; k_idx(i,j+1)]; V = [V; coeff];
            end

            % South / bottom boundary: y = -0.5 (Neumann 0)
            if i > 1
                ks = 2 * K(i,j) * K(i-1,j) / (K(i,j) + K(i-1,j));
                coeff = ks / (h^2);
                V_diag = V_diag - coeff;
                I = [I; row]; J = [J; k_idx(i-1,j)]; V = [V; coeff];
            end

            % North / top boundary: y = +0.5 (Neumann 0)
            if i < N
                kn = 2 * K(i,j) * K(i+1,j) / (K(i,j) + K(i+1,j));
                coeff = kn / (h^2);
                V_diag = V_diag - coeff;
                I = [I; row]; J = [J; k_idx(i+1,j)]; V = [V; coeff];
            end

            % Diagonal entry
            I = [I; row]; J = [J; row]; V = [V; V_diag];
        end
    end

    L = sparse(I, J, V, N2, N2);
end

function M = IBMatrix(X_grid, Y_grid, eps, eta, h, N)
    NL = length(eps);
    total_cells = N * N;

    row_idx = [];    col_idx = [];    weights = [];

    for k = 1:NL
        phi_x = abs(X_grid - eps(k));
        phi_y = abs(Y_grid - eta(k));

        % Identify grid nodes within the kernel support (1.5*h radius)
        nearby_indices = find(phi_x <= 1.5*h & phi_y <= 1.5*h);

        if ~isempty(nearby_indices)
            wx = Deltafunc(X_grid(nearby_indices) - eps(k), h);
            wy = Deltafunc(Y_grid(nearby_indices) - eta(k), h);

            % 2D kernel weight calculation: W = phi(x)*phi(y)*h^2
            % h^2 is the area element of the grid cell
            kernel_vals = (wx .* wy) * (h^2);

            row_idx = [row_idx; k * ones(length(nearby_indices), 1)];
            col_idx = [col_idx; nearby_indices];
            weights = [weights; kernel_vals];
        end
    end

    % Assemble the final sparse interpolation matrix
    % Dimensions: [Number of Lagrangian Points] x [Total Grid Cells]
    M = sparse(row_idx, col_idx, weights, NL, total_cells);
end

function phi = Deltafunc(r, h)
    phi = zeros(size(r));
    abs_rh = abs(r) / h;

    mask1 = (abs_rh <= 0.5);
    if any(mask1(:))
        phi(mask1) = (1/(3*h)) * (1 + sqrt(-3 * abs_rh(mask1).^2 + 1));
    end

    mask2 = (abs_rh > 0.5 & abs_rh <= 1.5);
    if any(mask2(:))
        phi(mask2) = (1/(6*h)) * ...
            (5 - 3 * abs_rh(mask2) - sqrt(-3 * (1 - abs_rh(mask2)).^2 + 1));
    end
end

function [stop_signal, error_val] = Monitor_Simulation(T, T_old, step, t_curr, ...
                                                       x_v, y_v, X_lag, Y_lag, ...
                                                       tolerance, log_interval, plot_interval)
    stop_signal = false;
    error_val = max(abs(T(:) - T_old(:)));

    if max(abs(T(:))) > 20 && t_curr > 1.0
        error('Simulation crashed: NaN detected or temperature exploded.');
    end

    if mod(step, log_interval) == 0
        fprintf('Step: %d | Time: %.6f s | Error: %.4e\n', step, t_curr, error_val);
    end

    if mod(step, plot_interval) == 0 || step == 0
        imagesc(x_v, y_v, T); 
        caxis([0 1]); 
        hold on;

        if max(T(:)) - min(T(:)) > (tolerance * 0.1)
            contour(x_v, y_v, T, 30, 'c', 'LineWidth', 0.5);
        end

        plot([X_lag; X_lag(1)], [Y_lag; Y_lag(1)], 'w-', 'LineWidth', 2);
        set(gca, 'YDir', 'normal'); colorbar; 
        colormap jet; axis([-0.5 0.5 -0.5 0.5]); axis equal;
        title(sprintf('Time: %.4f s | Error: %.2e | Step: %d', t_curr, error_val, step));
        xlabel('x'); ylabel('y'); drawnow; 
        hold off;
    end

    if error_val < tolerance && t_curr > 0.01
        stop_signal = true;
    end
end

function Plot_Snapshot(fig_num, x_vec, T_data, current_t, eps_L, eta_L, title_str)
    figure(fig_num);
    imagesc(x_vec, x_vec, T_data);
    caxis([0 1]);
    hold on;

    contour(x_vec, x_vec, T_data, 50, 'c', 'LineWidth', 0.5);
    plot([eps_L; eps_L(1)], [eta_L; eta_L(1)], 'w-', 'LineWidth', 2);

    set(gca, 'YDir', 'normal'); colorbar; colormap jet;
    axis equal; axis([-0.5 0.5 -0.5 0.5]);
    title(sprintf('%s %.4f s', title_str, current_t));
    xlabel('x'); ylabel('y');
    hold off;
    drawnow;
end

function N = N_from_grid(X_grid)
    N = size(X_grid, 1);
end