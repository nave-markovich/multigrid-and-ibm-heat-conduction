function NumericsQ3()
% =========================================================================
% Project: Advanced Numerical Methods - Question 3
% Task: BiCGStab with MG V-Cycle as a LEFT Preconditioner
% Method: FVM discretization, BiCGStab solver, 1 MG-V-Cycle per iteration
% =========================================================================
clc; clear ALL; close all;
tic;

% --- 1. Parameters ---
N = 200;            % Grid size (Consistent with Q1, Q2)
L = 1.0;            % Domain length
h = L/N;            % Mesh size
k1 = 1e-3;          % Background conductivity
k2 = 100.0;         % High conductivity (top-right)
tol = 1e-6;         % Convergence tolerance
maxIter = 100;      % Max iterations for BiCGStab

% --- 2. Grid & Material Setup ---
x = linspace(-0.5 + h/2, 0.5 - h/2, N);
[xc, yc] = meshgrid(x, x);
bc_func = @(y) 1 - 4 * y.^2;
% --- 3. Build Global Sparse Matrix A and RHS b ---
fprintf('Q3: Assembling Global Sparse Matrix (N=%d)...\n', N);
[A, b] = Build_Global_System(N, h, k1, k2, bc_func);

% --- 4. Setup Multigrid Hierarchy for Preconditioner ---

% --- System Configuration ---
dt = Inf; % Inf forces Steady-State equations (1/dt = 0)
% Dynamically calculate coarsest grid size (e.g., 3 levels of coarsening)
% For N=200 -> minN = 200 / 2^3 = 25. 
% For N=256 -> minN = 256 / 2^3 = 32.
num_coarsenings = 3;    
minN = N / (2^num_coarsenings);
mg_data = Setup_MG_Hierarchy(N, h, dt, k1, k2, minN);
%--------------------------------------%

% Define Preconditioner Handle (M^-1 * r)
M_inv = @(r) MG_Preconditioner_Step(r, mg_data);
% --- 5. Solve using BiCGStab ---
fprintf('Q3: Starting BiCGStab with MG Preconditioner...\n');
x0 = zeros(N*N, 1); % Initial guess (vectorized)
[T_vec, flag, relres, iter, resvec] = bicgstab(A, b, tol, maxIter, M_inv, [], x0);
Q3timer = toc;
if flag == 0
    fprintf('Converged in %d iterations. Final RelRes: %.2e\n', iter, relres);
else
    fprintf('BiCGStab failed. Flag: %d\n', flag);
end
fprintf('Total time: %.4f seconds\n', Q3timer);

% --- 6. Visualization ---
Plot_Results(T_vec, x, x, resvec, norm(b));

end

% =========================================================================
% SYSTEM ASSEMBLY & PRECONDITIONER FUNCTIONS
% =========================================================================

function [A, b] = Build_Global_System(N, h, k1, k2, bc_func)
    % 1. Get Physics Vectorized (Reuse existing logic)
    x = linspace(-0.5+h/2, 0.5-h/2, N);
    [xc, yc] = meshgrid(x, x);
    K = k1 * ones(N, N); 
    K(xc >= 0 & yc >= 0) = k2;
    
    % Reuse Build_Coeffs for consistency (Steady State: dt = Inf)
    [aP, aW, aE, aS, aN] = Build_Coeffs(K, h, Inf);
    
    % 2. Build RHS vector (b) - Vectorized Boundary Conditions
    b_mat = zeros(N, N);
    % Dirichlet on West (i=1) and East (i=N)
    b_mat(:, 1) = b_mat(:, 1) + aW(:, 1) .* bc_func(x'); % x' is y-coords in meshgrid
    b_mat(:, N) = b_mat(:, N) + aE(:, N) .* bc_func(x');
    % Neumann on South/North is already handled by zeroing aS/aN in Build_Coeffs
    b = b_mat(:);

    % 3. Build Sparse Matrix A - Fully Vectorized
    idx = reshape(1:N*N, N, N);
    
    % Center Diagonal
    ii = idx(:); jj = idx(:); ss = aP(:);
    
    % West Neighbors (i > 1)
    ii_w = idx(:, 2:end); jj_w = idx(:, 1:end-1); ss_w = -aW(:, 2:end);
    % East Neighbors (i < N)
    ii_e = idx(:, 1:end-1); jj_e = idx(:, 2:end); ss_e = -aE(:, 1:end-1);
    % South Neighbors (j > 1)
    ii_s = idx(2:end, :); jj_s = idx(1:end-1, :); ss_s = -aS(2:end, :);
    % North Neighbors (j < N)
    ii_n = idx(1:end-1, :); jj_n = idx(2:end, :); ss_n = -aN(1:end-1, :);
    
    % Assemble all at once
    I = [ii; ii_w(:); ii_e(:); ii_s(:); ii_n(:)];
    J = [jj; jj_w(:); jj_e(:); jj_s(:); jj_n(:)];
    S = [ss; ss_w(:); ss_e(:); ss_s(:); ss_n(:)];
    
    A = sparse(I, J, S, N*N, N*N);
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

function Rc = Restrict(Rf)
    % 2x2 Cell averaging
    Rc = 0.25*(Rf(1:2:end,1:2:end)+Rf(2:2:end,1:2:end)+Rf(1:2:end,2:2:end)+Rf(2:2:end,2:2:end)); 
end

function Ef = Prolong(Ec)
    % Piecewise constant expansion (Coarse to Fine)
    % Fully vectorized using repelem for O(N) performance
    Ef = repelem(Ec, 2, 2);
end

% =========================================================================
% PRECONDITIONER & MULTIGRID CORE (Modernized for aP, aW, aE, aS, aN)
% =========================================================================

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
% Recursive Multigrid V-Cycle Algorithm
% U: Current guess (or error correction)
% b: Right Hand Side (or residual from the finer level)
% lvl: Current level in the multigrid hierarchy

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

function Plot_Results(T_vec, x, y, res_history, b_norm)
    N = length(x);
    T_final = reshape(T_vec, N, N);
    [xc, yc] = meshgrid(x, y);

    % --- Figure 3: Temperature Map ---
    figure(3); clf; hold on;
    % Displaying the temperature field using filled contours
    contourf(xc, yc, T_final, 40, 'LineColor','none'); 
    colorbar; colormap jet; axis equal;
    title('Question 3: Temperature Distribution (BiCGStab + MG V-Cycle)');
    xlabel('x'); ylabel('y');
    
    % Highlighting the material property interfaces at x=0 and y=0
    plot([0 0], [-0.5 0.5], 'k--', 'LineWidth', 1.2); 
    plot([-0.5 0.5], [0 0], 'k--', 'LineWidth', 1.2); 
    hold off;

    % --- Figure 33: Convergence History ---
    figure(33); clf;
    % Plotting the relative residual history to verify solver convergence
    semilogy(0:length(res_history)-1, res_history / b_norm, '-bo', 'LineWidth', 1.5, 'MarkerSize', 4);
    grid on;
    title('BiCGStab Convergence History (MG Preconditioned)');
    xlabel('Iteration');
    ylabel('Relative Residual Norm: ||b - Ax|| / ||b||');
end