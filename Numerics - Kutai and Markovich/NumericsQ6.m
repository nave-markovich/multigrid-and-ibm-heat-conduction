function NumericsQ6()
% =========================================================================
% Project: Advanced Numerical Methods - Question 6
% Task: Solve the unsteady 2D heat equation with an immersed circular hole.
% Method: Fully implicit direct-forcing Immersed Boundary Method (IBM) 
%         using a reduced Schur complement formulation.
% =========================================================================
clc; close all;
tic;

%% 1. Parameters Definition
N  = 200;                  % Structured grid size
L  = 1.0;                  % Domain length
h  = L / N;                % Uniform cell size
dt = 1e-3;                 % Fully implicit time step
k1 = 1e-3;                 % Conductivity in most of the plate
k2 = 100.0;                % Conductivity in top-right quadrant
R_hole = 0.2;              % Hole radius
T_hole = 2.0;              % Dirichlet temperature on hole boundary
plot_times = [1.0, 5.0];   % Requested output times
ss_tol     = 1e-6;         % Steady-state tolerance
t_max_ss   = 500.0;        % Maximum simulation time limit

%% 2. Initialization and Assembly (Using Local Functions)
% Setup Grid and Conductivity Field
[x, y, K] = Setup_Grid_And_Conductivity(N, h, k1, k2);

% Assemble the Eulerian Operator (A) and Boundary Conditions (RHS_bc)
[A, RHS_bc] = Assemble_Eulerian_Operator(N, h, dt, K, x, y);

% Assemble the IBM Operators (Interpolation Q and Spreading B_tilde)
[Q, B_tilde, M, T_hole_vec] = Assemble_IBM_Operators(N, h, x, y, R_hole, T_hole);

% Precompute LU factorizations for the Eulerian operator and Schur matrix
[L_A, U_A, P_A, Q_A, invA_B, L_S, U_S, P_S] = Precompute_Schur(A, Q, B_tilde);

%% 3. Unsteady Time Integration Loop
fprintf('Starting time integration...\n');
T_old = zeros(N*N, 1);
t = 0;
next_plot = 1;

while true
    t = t + dt;
    
    % Step 1: Predictor RHS
    RHS_star = ((h^2) / dt) * T_old + RHS_bc;
    
    % Solve for the intermediate temperature field
    invA_RHS = Q_A * (U_A \ (L_A \ (P_A * RHS_star)));
    
    % Step 2: Reduced Schur solve
    RHS_S = T_hole_vec - Q * invA_RHS;
    lambda = U_S \ (L_S \ (P_S * RHS_S));
    
    % Step 3: Corrected temperature
    T_new = invA_RHS + invA_B * lambda;
    
    % Convergence measure
    diff = max(abs(T_new - T_old));
    T_old = T_new;
    
    % Plot at requested times
    if next_plot <= length(plot_times) && t >= plot_times(next_plot) - 1e-8
        Plot_Results(T_new, x, y, R_hole, sprintf('Temperature Field at t = %.1f', t));
        fprintf('Time t = %.1f reached.\n', t);
        next_plot = next_plot + 1;
    end
    
    % Check steady-state stop condition (only allow after t >= 5)
    if t >= plot_times(end) && diff < ss_tol
        Plot_Results(T_new, x, y, R_hole, 'Temperature Field at Steady State');
        fprintf('Steady state reached at t = %.3f\n', t);
        break;
    end
    
    % Safety exit to prevent infinite loops
    if t > t_max_ss
        fprintf('Max time reached without hitting steady state tolerance.\n');
        break;
    end
end
toc;
end

% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function [x, y, K] = Setup_Grid_And_Conductivity(N, h, k1, k2)
    % Initialize 1D grid coordinates
    x = linspace(-0.5 + h/2, 0.5 - h/2, N);
    y = linspace(-0.5 + h/2, 0.5 - h/2, N);
    
    % Create 2D coordinate grids for vectorized logical indexing
    [X_grid, Y_grid] = meshgrid(x, y);
    
    % Initialize conductivity field with the background value (k1)
    K = k1 * ones(N, N);
    
    % Vectorized assignment: Assign k2 to the top-right quadrant
    K(X_grid > 0 & Y_grid > 0) = k2;
end

function [A, RHS_bc] = Assemble_Eulerian_Operator(N, h, dt, K, x, y)
    fprintf('Assembling fine-grid Eulerian operator...\n');
    num_elements = 5 * N * N;
    A_i = zeros(num_elements, 1);
    A_j = zeros(num_elements, 1);
    A_v = zeros(num_elements, 1);
    RHS_bc = zeros(N*N, 1);
    idx_cnt = 0;
    
    for i = 1:N       % x index (columns)
        for j = 1:N   % y index (rows)
            row = j + (i-1)*N;   % Column-major indexing
            
            diag_val = (h^2) / dt;   % Time derivative contribution
            rhs_val  = 0;            % Boundary contribution
            
            % East face
            if i < N
                k_E = 2 * K(j,i) * K(j,i+1) / (K(j,i) + K(j,i+1));
                idx_cnt = idx_cnt + 1;
                A_i(idx_cnt) = row; A_j(idx_cnt) = row + N; A_v(idx_cnt) = -k_E;
                diag_val = diag_val + k_E;
            else
                % Dirichlet at x = 0.5
                k_E = 2 * K(j,i);
                diag_val = diag_val + k_E;
                rhs_val = rhs_val + k_E * (1 - 4*y(j)^2);
            end
            
            % West face
            if i > 1
                k_W = 2 * K(j,i) * K(j,i-1) / (K(j,i) + K(j,i-1));
                idx_cnt = idx_cnt + 1;
                A_i(idx_cnt) = row; A_j(idx_cnt) = row - N; A_v(idx_cnt) = -k_W;
                diag_val = diag_val + k_W;
            else
                % Dirichlet at x = -0.5
                k_W = 2 * K(j,i);
                diag_val = diag_val + k_W;
                rhs_val = rhs_val + k_W * (1 - 4*y(j)^2);
            end
            
            % North face
            if j < N
                k_N = 2 * K(j,i) * K(j+1,i) / (K(j,i) + K(j+1,i));
                idx_cnt = idx_cnt + 1;
                A_i(idx_cnt) = row; A_j(idx_cnt) = row + 1; A_v(idx_cnt) = -k_N;
                diag_val = diag_val + k_N;
            else
                % Neumann at y = 0.5, zero flux
            end
            
            % South face
            if j > 1
                k_S = 2 * K(j,i) * K(j-1,i) / (K(j,i) + K(j-1,i));
                idx_cnt = idx_cnt + 1;
                A_i(idx_cnt) = row; A_j(idx_cnt) = row - 1; A_v(idx_cnt) = -k_S;
                diag_val = diag_val + k_S;
            else
                % Neumann at y = -0.5, zero flux
            end
            
            % Diagonal
            idx_cnt = idx_cnt + 1;
            A_i(idx_cnt) = row; A_j(idx_cnt) = row; A_v(idx_cnt) = diag_val;
            RHS_bc(row) = rhs_val;
        end
    end
    A = sparse(A_i(1:idx_cnt), A_j(1:idx_cnt), A_v(1:idx_cnt), N*N, N*N);
end

function [Q, B_tilde, M, T_hole_vec] = Assemble_IBM_Operators(N, h, x, y, R_hole, T_hole)
    fprintf('Assembling IBM Operators...\n');
    M = round(2 * pi * R_hole / h);   % Number of Lagrangian points
    dtheta = 2 * pi / M;
    theta = (0:M-1)' * dtheta;
    x_lag = R_hole * cos(theta);
    y_lag = R_hole * sin(theta);
    ds = 2 * pi * R_hole / M;
    
    Q_i = []; Q_j = []; Q_v = [];
    B_i = []; B_j = []; B_v = [];
    
    for k = 1:M
        xl = x_lag(k);
        yl = y_lag(k);
        
        % Find Eulerian points within kernel support
        i_min = max(1, floor((xl - (-0.5)) / h) - 2);
        i_max = min(N, ceil((xl - (-0.5)) / h) + 2);
        j_min = max(1, floor((yl - (-0.5)) / h) - 2);
        j_max = min(N, ceil((yl - (-0.5)) / h) + 2);
        
        for i = i_min:i_max
            for j = j_min:j_max
                rx = (x(i) - xl) / h;
                ry = (y(j) - yl) / h;
                w = Peskin_Kernel(rx) * Peskin_Kernel(ry);
                
                if w > 0
                    row = j + (i-1)*N;
                    
                    % Interpolation operator
                    Q_i(end+1) = k;
                    Q_j(end+1) = row;
                    Q_v(end+1) = w;
                    
                    % Spreading operator
                    B_i(end+1) = row;
                    B_j(end+1) = k;
                    B_v(end+1) = w * ds;
                end
            end
        end
    end
    Q = sparse(Q_i, Q_j, Q_v, M, N*N);
    B_tilde = sparse(B_i, B_j, B_v, N*N, M);
    T_hole_vec = T_hole * ones(M, 1);
end

function [L_A, U_A, P_A, Q_A, invA_B, L_S, U_S, P_S] = Precompute_Schur(A, Q, B_tilde)
    fprintf('Precomputing LU factorizations to speed up time-stepping...\n');
    % Use 4-output sparse LU for A to minimize fill-in
    [L_A, U_A, P_A, Q_A] = lu(A);
    
    % Compute A^{-1} * B_tilde once
    invA_B = Q_A * (U_A \ (L_A \ (P_A * B_tilde)));
    
    % Compute reduced Schur matrix
    S = Q * invA_B;
    
    % LU factorization of reduced Schur system
    [L_S, U_S, P_S] = lu(S);
end

function val = Peskin_Kernel(r)
    % Peskin 4-point discrete delta kernel
    ar = abs(r);
    if ar <= 1
        val = (1/8) * (3 - 2*ar + sqrt(max(0, 1 + 4*ar - 4*ar^2)));
    elseif ar <= 2
        val = (1/8) * (5 - 2*ar - sqrt(max(0, -7 + 12*ar - 4*ar^2)));
    else
        val = 0;
    end
end

function Plot_Results(T_vec, x, y, R_hole, title_str)
    % Reshape perfectly aligns with Column-Major ordering (x columns, y rows)
    N = length(x);
    T_mat = reshape(T_vec, N, N);
    [X, Y] = meshgrid(x, y);
    
    figure;
    contourf(X, Y, T_mat, 20, 'LineColor', 'none');
    colorbar; colormap jet;
    caxis([0, 2]);
    hold on;
    
    % White circle for hole visualization
    theta = linspace(0, 2*pi, 100);
    plot(R_hole*cos(theta), R_hole*sin(theta), 'w-', 'LineWidth', 2);
    
    title(title_str);
    xlabel('x');
    ylabel('y');
    axis equal;
    axis([-0.5 0.5 -0.5 0.5]);
    drawnow;
end