function NumericsQ4()
% =========================================================================
% Project: Advanced Numerical Methods - Question 4
% Task: Compare computational times and estimate complexity
% 
% Methodology:
% 1. Create a highly accurate reference solution (1e-12 tolerance).
% 2. Calibrate BiCGStab to find the exact residual tolerance needed to 
%    reach 6-decimal true accuracy (||T - T_ref||_inf < 5e-7).
% 3. Calibrate Pure MG to find the exact number of V-cycles needed to 
%    reach the exact same true accuracy.
% 4. Time both methods "blindly" using their calibrated parameters to
%    ensure a perfectly fair performance comparison without the overhead 
%    of error-checking inside the timing loop.
% =========================================================================

clc; close all; clear All;

% --- 1. Parameters ---
k1 = 1e-3;
k2 = 100.0;
dt = Inf;                    % Steady state problem
minN = 16;                   % Coarsest grid size in the MG hierarchy
bc_func = @(y) 1 - 4*y.^2;

% Using powers of 2 for perfect grid halving
N_list = [128, 256, 512, 1024];

% Accuracy settings for 6 decimal digits
acc_tol = 5e-7;              
ref_tol = 1e-12;             % Tight tolerance for the reference solution
maxMG   = 500;               
maxBiCG = 500;               
nRuns   = 3;                 % Number of runs for averaging to reduce OS noise

times_MG   = nan(size(N_list));
times_BiCG = nan(size(N_list));

fprintf('============================================================\n');
fprintf('Q4: Runtime comparison for Items 2 and 3\n');
fprintf('Target accuracy: ||T - T_ref||_inf < %.1e\n', acc_tol);
fprintf('============================================================\n\n');

for k = 1:length(N_list)
    N = N_list(k);
    h = 1.0 / N;
    fprintf('Testing Grid Size N = %d\n', N);
    
    % --- Step A: Build reference solution ---
    mg_data = Setup_MG_Hierarchy(N, h, dt, k1, k2, minN);
    b_mat   = Build_RHS(mg_data(1), bc_func);
    A_fine  = Build_Sparse_A(mg_data(1));
    M_fun   = @(r) MG_Preconditioner_Step(r, mg_data);
    
    fprintf('   Building reference solution (tol = 1e-12)... ');
    x0 = zeros(N*N,1);
    [x_ref, flag_ref] = bicgstab(A_fine, b_mat(:), ref_tol, 2000, M_fun, [], x0);
    if flag_ref ~= 0
        error('Reference solve failed for N = %d.', N);
    end
    T_ref = reshape(x_ref, N, N);
    fprintf('Done.\n');
    
    % --- Step B: Calibrate BiCGStab Tolerance ---
    fprintf('   Calibrating BiCGStab tolerance ... ');
    tol_bicg = Find_BiCG_Tolerance(A_fine, b_mat(:), mg_data, T_ref, acc_tol, maxBiCG);
    fprintf('tol = %.1e\n', tol_bicg);
    
    % --- Step C: Calibrate Pure MG Cycles ---
    fprintf('   Calibrating Pure MG cycles ... ');
    U_calib = zeros(N, N);
    req_cycles_MG = 0;
    for cycle = 1:maxMG
        U_calib = V_Cycle_Recursive(U_calib, b_mat, mg_data, 1);
        if max(abs(U_calib(:) - T_ref(:))) < acc_tol
            req_cycles_MG = cycle;
            break;
        end
    end
    if req_cycles_MG == 0
        error('Pure MG failed to reach accuracy within %d cycles.', maxMG);
    end
    fprintf('%d cycles required\n', req_cycles_MG);
    
    % --- Step D: Time Item 2 (Pure MG) ---
    % Notice how the timing loop is completely free of error checking!
    mg_runs = nan(1, nRuns);
    for r = 1:nRuns
        U_solve = zeros(N, N);
        tic; % Start Timing 
        for cycle = 1:req_cycles_MG
            U_solve = V_Cycle_Recursive(U_solve, b_mat, mg_data, 1);
        end
        mg_runs(r) = toc; % Stop Timing
    end
    times_MG(k) = mean(mg_runs);
    fprintf('   Pure MG average time     = %.6f s\n', times_MG(k));
    
    % --- Step E: Time Item 3 (BiCGStab + MG) ---
    bicg_runs = nan(1, nRuns);
    for r = 1:nRuns
        tic;
        [x, flag_b] = bicgstab(A_fine, b_mat(:), tol_bicg, maxBiCG, M_fun, [], x0);
        t_run = toc;

        if flag_b ~= 0
            bicg_runs(r) = nan;
            warning('BiCGStab failed during timing loop.');
            break;
        end

        if max(abs(x - T_ref(:))) >= acc_tol
            bicg_runs(r) = nan;
            warning('BiCGStab did not reach the required 6-digit accuracy during timing loop.');
            break;
        end

        bicg_runs(r) = t_run;
    end

    times_BiCG(k) = mean(bicg_runs, 'omitnan');
    fprintf('   BiCGStab+MG average time = %.6f s\n', times_BiCG(k));
    
    fprintf('------------------------------------------------------------\n');
end

% --- Step 4: Complexity Estimation & Plotting ---
valid_MG = isfinite(times_MG);
valid_BiCG = isfinite(times_BiCG);

beta_MG = polyfit(log(N_list(valid_MG)), log(times_MG(valid_MG)), 1);
beta_BiCG = polyfit(log(N_list(valid_BiCG)), log(times_BiCG(valid_BiCG)), 1);

fprintf('\n============================================================\n');
fprintf('Summary Table\n');
fprintf('============================================================\n');
fprintf('%-8s | %-18s | %-18s\n', 'N', 't_MG [s]', 't_BiCG+MG [s]');
for k = 1:length(N_list)
    fprintf('%-8d | %-18.6f | %-18.6f\n', N_list(k), times_MG(k), times_BiCG(k));
end
fprintf('------------------------------------------------------------\n');
fprintf('Estimated Complexity (O(N^beta)):\n');
fprintf('Pure MG Beta      : %.4f\n', beta_MG(1));
fprintf('BiCGStab+MG Beta  : %.4f\n', beta_BiCG(1));
fprintf('============================================================\n');

% Plot runtime comparison
figure('Name', 'Q4: Computational Complexity', 'Color', 'w');
loglog(N_list(valid_MG), times_MG(valid_MG), '-or', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r'); hold on;
loglog(N_list(valid_BiCG), times_BiCG(valid_BiCG), '-sb', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
grid on; grid minor; set(gca, 'FontSize', 12);
xlabel('Grid Size (N)'); ylabel('Computational Time [sec]');
title('Q4: Runtime Comparison and Complexity');
legend(sprintf('Item 2: Pure MG (\\beta \\approx %.2f)', beta_MG(1)), ...
       sprintf('Item 3: BiCGStab+MG (\\beta \\approx %.2f)', beta_BiCG(1)), 'Location', 'NorthWest');

end

% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function b_mat = Build_RHS(lvl, bc_func)
    % Evaluates Dirichlet boundary conditions and builds the RHS matrix
    N = lvl.N;
    h = lvl.h;
    y = linspace(-0.5 + h/2, 0.5 - h/2, N)';
    T_bc = bc_func(y);
    b_mat = zeros(N, N);
    
    % Dirichlet contribution at x = -0.5 and x = +0.5
    b_mat(:,1)   = b_mat(:,1)   + lvl.aW(:,1)   .* T_bc;
    b_mat(:,end) = b_mat(:,end) + lvl.aE(:,end) .* T_bc;
end

function A = Build_Sparse_A(lvl)
    % Fully vectorized sparse assembly from the 5-point stencil arrays
    N = lvl.N;
    idx = reshape(1:N*N, N, N);

    % Main diagonal
    ii = idx(:);
    jj = idx(:);
    ss = lvl.aP(:);

    % West neighbors
    ii_w = idx(:, 2:end);
    jj_w = idx(:, 1:end-1);
    ss_w = -lvl.aW(:, 2:end);

    % East neighbors
    ii_e = idx(:, 1:end-1);
    jj_e = idx(:, 2:end);
    ss_e = -lvl.aE(:, 1:end-1);

    % South neighbors
    ii_s = idx(2:end, :);
    jj_s = idx(1:end-1, :);
    ss_s = -lvl.aS(2:end, :);

    % North neighbors
    ii_n = idx(1:end-1, :);
    jj_n = idx(2:end, :);
    ss_n = -lvl.aN(1:end-1, :);

    % Assemble all entries in one call
    I = [ii; ii_w(:); ii_e(:); ii_s(:); ii_n(:)];
    J = [jj; jj_w(:); jj_e(:); jj_s(:); jj_n(:)];
    S = [ss; ss_w(:); ss_e(:); ss_s(:); ss_n(:)];

    A = sparse(I, J, S, N*N, N*N);
end

function [aP, aW, aE, aS, aN] = Build_Coeffs(K, h, dt)
    % Finite Volume coefficients with harmonic mean conductivity
    N = size(K, 1);
    harm = @(a,b) 2*a.*b ./ (a + b + eps);
    Kx = harm(K(:,1:end-1), K(:,2:end));
    Ky = harm(K(1:end-1,:), K(2:end,:));
    invh2 = 1 / (h*h);
    
    aW = zeros(N, N); aE = zeros(N, N); aS = zeros(N, N); aN = zeros(N, N);
    aW(:,1) = (2*K(:,1)) * invh2; aW(:,2:end) = Kx * invh2;
    aE(:,1:end-1) = Kx * invh2; aE(:,end) = (2*K(:,end)) * invh2;
    aS(2:end,:) = Ky * invh2;
    aN(1:end-1,:) = Ky * invh2;
    
    % Diagonal includes unsteady term and neighbors
    aP = (1/dt) + aW + aE + aS + aN;
end


function U = V_Cycle_Recursive(U, b, mg_data, lvl)
%     % Recursive Multigrid V-Cycle Algorithm
%     % U: Current guess (or error correction)
%     % b: Right Hand Side (or residual from the finer level)
%     % lvl: Current level in the multigrid hierarchy

% Base Case: If we reached the coarsest level, solve exactly using direct inversion
if lvl == numel(mg_data)
    U(:) = mg_data(lvl).A_coarse \ b(:);
    return;
end

% 1. Pre-Smoothing: Reduce high-frequency errors using Line Gauss-Seidel
U = Smooth_Line_GS(U, b, mg_data(lvl));

% 2. Compute Residual: Matrix-free calculation (res = b - A*U)
res = b - Apply_A_Mat(U, mg_data(lvl));

% 3. Restriction: Transfer the residual to the coarser grid
% (Requires the external 'Restrict.m' function)
r_coarse = Restrict(res);

% 4. Recursive Call: Solve the error equation (A*e = r) on the coarser grid
e_coarse = V_Cycle_Recursive(zeros(size(r_coarse)), r_coarse, mg_data, lvl + 1);

% 5. Prolongation & Correction: Interpolate the error back to the fine grid and add it
% (Requires the external 'Prolong.m' function)
U = U + Prolong(e_coarse);

% 6. Post-Smoothing: Reduce any new high-frequency errors introduced by prolongation
U = Smooth_Line_GS(U, b, mg_data(lvl));
end


function Rc = Restrict(Rf)
    % 2x2 Cell averaging
    Rc = 0.25*(Rf(1:2:end,1:2:end)+Rf(2:2:end,1:2:end)+Rf(1:2:end,2:2:end)+Rf(2:2:end,2:2:end)); 
end

function Ef = Prolong(Ec)
    % Piecewise constant expansion (Coarse to Fine)
    % Fully vectorized using repelem for O(N) performance
    Ef = repelem(Ec, 2, 2);
end

function AU = Apply_A_Mat(U, lvl)
    % Matrix-Free Matrix-Vector Multiplication (A * U)
    % Calculates the Laplacian operator directly on the 2D grid without building 
    % the global sparse matrix A. This drastically reduces memory usage and speeds up computation.
    
    % Shift the matrix U to get neighbor values (padded with zeros at the boundaries)
    Uw = [zeros(size(U,1),1), U(:,1:end-1)]; % West neighbors
    Ue = [U(:,2:end), zeros(size(U,1),1)];   % East neighbors
    Us = [zeros(1,size(U,2)); U(1:end-1,:)]; % South neighbors
    Un = [U(2:end,:); zeros(1,size(U,2))];   % North neighbors
    
    % Apply the Finite Volume stencil using precomputed coefficients
    AU = lvl.aP.*U - lvl.aW.*Uw - lvl.aE.*Ue - lvl.aS.*Us - lvl.aN.*Un;
end

function U = Smooth_Line_GS(U, b, lvl_data)
    % Alternating Direction Line Gauss-Seidel Smoother
    % Sweeps first along the X-axis (columns), then along the Y-axis (rows)
    U = Sweep_Lines(U, b, lvl_data, 'X');
    U = Sweep_Lines(U, b, lvl_data, 'Y');
end

function U = Sweep_Lines(U, b, lvl, direction)
    % Solves tridiagonal systems for lines of cells implicitly.
    % Uses the pre-calculated finite volume coefficients to avoid redundant physics calculations.
    
    N = lvl.N;
    aP = lvl.aP; aW = lvl.aW; aE = lvl.aE; aS = lvl.aS; aN = lvl.aN;
    
    if strcmp(direction, 'X')
        % Sweep X: Solve each horizontal row (j) implicitly
        for j = 1:N
            % Extract tridiagonal diagonals for the current row
            diag_v = aP(j, :); 
            sub = [0, -aW(j, 2:end)]; 
            sup = [-aE(j, 1:end-1), 0];
            
            rhs = b(j, :);
            % Add known (explicit) neighboring values from South and North
            if j > 1, rhs = rhs + aS(j, :) .* U(j-1, :); end
            if j < N, rhs = rhs + aN(j, :) .* U(j+1, :); end
            
            % Solve the 1D tridiagonal system using the Thomas algorithm
            U(j, :) = thomas_solver(sub, diag_v, sup, rhs);
        end
    else
        % Sweep Y: Solve each vertical column (i) implicitly
        for i = 1:N
            % Extract tridiagonal diagonals for the current column
            diag_v = aP(:, i); 
            sub = [0; -aS(2:end, i)]; 
            sup = [-aN(1:end-1, i); 0];
            
            rhs = b(:, i);
            % Add known (explicit) neighboring values from West and East
            if i > 1, rhs = rhs + aW(:, i) .* U(:, i-1); end
            if i < N, rhs = rhs + aE(:, i) .* U(:, i+1); end
            
            % Solve the 1D tridiagonal system using the Thomas algorithm
            U(:, i) = thomas_solver(sub, diag_v, sup, rhs);
        end
    end
end

function x = thomas_solver(a, b, c, d)
    % Robust O(N) solver for tridiagonal systems
    % Force column vectors to prevent dimension mismatch in Y-sweeps
    a = a(:); b = b(:); c = c(:); d = d(:);
    
    n = length(b);
    for i = 2:n
        m = a(i)/b(i-1);
        b(i) = b(i) - m*c(i-1);
        d(i) = d(i) - m*d(i-1);
    end
    
    x = zeros(n, 1);
    x(n) = d(n)/b(n);
    for i = n-1:-1:1
        x(i) = (d(i) - c(i)*x(i+1))/b(i);
    end
end

function mg_data = Setup_MG_Hierarchy(N_fine, h_fine, dt, k1, k2, minN)
    lvl = 1; curN = N_fine; curH = h_fine;
    while true
        mg_data(lvl).N = curN; 
        mg_data(lvl).h = curH;
        coords = (-0.5 + curH/2) + (0:curN-1)*curH;
        mg_data(lvl).x = coords; mg_data(lvl).y = coords;
        
        K = k1 * ones(curN, curN);
        K(coords >= 0, coords >= 0) = k2; 
        
        % Transferring dt to the coefficient function
        [aP, aW, aE, aS, aN] = Build_Coeffs(K, curH, dt);
        mg_data(lvl).aP = aP; mg_data(lvl).aW = aW; 
        mg_data(lvl).aE = aE; mg_data(lvl).aS = aS; mg_data(lvl).aN = aN;
        
        if curN <= minN || mod(curN, 2) ~= 0
            % Stops if we reach the minimum or if we can no longer divide by 2
            mg_data(lvl).A_coarse = Build_Sparse_A(mg_data(lvl));
            break;
        end

        curN = curN / 2; curH = curH * 2; lvl = lvl + 1;
    end
end

function z = MG_Preconditioner_Step(r_vec, mg_data)
    % This function serves as the Left Preconditioner for the BiCGStab solver.
    % It takes the 1D residual vector (r_vec), converts it to a 2D grid,
    % performs exactly ONE Multigrid V-Cycle to approximate the error correction,
    % and returns the flattened 1D correction vector (z).
    
    N = mg_data(1).N;
    r_mat = reshape(r_vec, N, N); % Convert 1D residual to 2D matrix
    
    % Initial guess for the error correction is zero. Run 1 V-Cycle.
    z_mat = V_Cycle_Recursive(zeros(N, N), r_mat, mg_data, 1);
    
    z = z_mat(:); % Flatten back to 1D vector for BiCGStab compatibility
end

function tol_good = Find_BiCG_Tolerance(A, b, mg_data, T_ref, acc_tol, maxBiCG)
    % Sweeps through a list of tolerances to find the most relaxed one that
    % strictly satisfies the true absolute error requirement.
    tol_list = [1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8, 1e-9, 1e-10];
    M_fun = @(rhs) MG_Preconditioner_Step(rhs, mg_data);
    tol_good = [];
    
    for q = 1:length(tol_list)
        tol = tol_list(q);
        [x, flag] = bicgstab(A, b, tol, maxBiCG, M_fun, [], zeros(size(b)));
        if flag ~= 0
            continue;
        end
        err_inf = max(abs(x - T_ref(:)));
        if err_inf < acc_tol
            tol_good = tol;
            return;
        end
    end
    error('Could not find a BiCGStab tolerance that reaches 6-decimal accuracy.');
end