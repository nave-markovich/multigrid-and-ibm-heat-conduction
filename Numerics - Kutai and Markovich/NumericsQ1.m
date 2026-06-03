function NumericsQ1()
% =========================================================================
% Project: Advanced Numerical Methods - Question 1
% Question 1 - Steady State Heat Equation
% Method: Gauss-Seidel by-lines (Sweeps in the X-direction)
% Discretization: Finite Volume Method with Harmonic Mean Conductivities
% =========================================================================
clc; clear ALL; close all;
tic;

% --- 1. Parameters ---
N = 200;            % Number of cells in each direction (200x200 grid)
L = 1.0;            % Domain length (unit square)
h = L/N;            % Mesh size (cell width)
k1 = 1e-3;          % Low-conductivity material
k2 = 100.0;         % High-conductivity material in the top-right quadrant
tol = 1e-6;         % Convergence tolerance for the iterative solver
maxIter = 20000;    % Safety limit for maximum iterations

% --- Boundary Condition Function Handles ---
bc_dirichlet = @(y) 1 - 4 * y.^2;    % Dirichlet BC for Left/Right sides
bc_neumann_S = @(x) 0 * x;           % Neumann BC (dT/dy) for South boundary
bc_neumann_N = @(x) 0 * x;           % Neumann BC (dT/dy) for North boundary


% --- 2. Grid & Conductivity (Material Property Map) ---
% Creating a cell-centered grid
x = linspace(-0.5 + h/2, 0.5 - h/2, N);
y = linspace(-0.5 + h/2, 0.5 - h/2, N);
[xc, yc] = meshgrid(x, y);

% Define Conductivity Matrix K: default to k1
K = k1 * ones(N, N);
% Set top-right quadrant (x >= 0 and y >= 0) to k2
K(xc >= 0 & yc >= 0) = k2;

% Precompute Harmonic Mean at cell interfaces (Required for flux continuity)
% harm = (2 * ka * kb) / (ka + kb)
harm = @(a,b) 2*a.*b./(a+b+eps);


% Scale interface conductivities by 1/h^2 to pre-calculate discrete Laplacian coefficients.
% This accounts for the gradient approximation (1/h) and normalization by cell area (1/h).
% Pre-calculating this once outside the loop optimizes performance by avoiding redundant divisions.
% Kx: Interface conductivities between horizontal neighbors (N x N-1)
Kx = harm(K(:, 1:end-1), K(:, 2:end)) / h^2; 
% Ky: Interface conductivities between vertical neighbors (N-1 x N)
Ky = harm(K(1:end-1, :), K(2:end, :)) / h^2; 

% --- 3. Boundary Conditions (BCs) ---
% Left/Right (x=-0.5, x=0.5): Dirichlet BC based on the profile 1 - 4*y^2
bcL = bc_dirichlet(yc(:,1));
bcR = bc_dirichlet(yc(:,end));

% Initialize Temperature field T
T = zeros(N, N);

% Initial guess: Linear interpolation between boundaries for faster convergence
for i = 1:N
    weight = (i-1)/(N-1);
    T(:,i) = (1-weight)*bcL + weight*bcR;
end

res_history = []; % Store error history for convergence plot
fprintf('Q1: Starting Gauss-Seidel by-lines (X-Sweep) (N=%d)\n', N);

% --- 4. Main Solver Loop (Gauss-Seidel) ---
for it = 1:maxIter
    Told = T;
    
    % SWEEP IN X DIRECTION: Solving Vertical Columns
    % We iterate from column i=1 to N, solving for all j-nodes in that column.
    for i = 1:N
        % 4a. Identify horizontal neighbor coefficients and values.
        % the wall is h/2 away from the boundary cell center, so the boundary
        % contribution gives an effective coefficient of 2*k/h^2.
        if i == 1 % Left Boundary Column
            kw = 2*K(:,i)/h^2; ke = Kx(:,i);
            Tw = bcL;          Te = T(:,i+1);
        elseif i == N % Right Boundary Column
            kw = Kx(:,i-1);    ke = 2*K(:,i)/h^2;
            Tw = T(:,i-1);     Te = bcR;
        else % Interior Columns
            kw = Kx(:,i-1);    ke = Kx(:,i);
            Tw = T(:,i-1);     Te = T(:,i+1);
        end
        
        % 4b. Identify vertical neighbor coefficients (North/South)
        % Neumann BC (insulated) is handled by setting interface k to 0 at edges
        ks = [0; Ky(:,i)]; % Southern neighbors
        kn = [Ky(:,i); 0]; % Northern neighbors
        
        % 4c. Construct the Tridiagonal System Components: a*T(j-1) + b*T(j) + c*T(j+1) = d
        % Vectorized calculation for the entire column i
        b = -(kw + ke + ks + kn); % Main diagonal (Negative sum of neighbors)
        a = ks;                   % Sub-diagonal
        c = kn;                   % Super-diagonal
        
        % Right-Hand Side (RHS): Contribution from known West/East neighbors
        d = -(kw.*Tw + ke.*Te); 
        d(1)   = d(1)   + (K(1,i)/h)   * bc_neumann_S(x(i)); % South flux
        d(end) = d(end) - (K(end,i)/h) * bc_neumann_N(x(i)); % North flux


        % 4d. Solve the column system using the Thomas Algorithm
        T(:,i) = thomas_solver(a, b, c, d);
    end
    
    % 4e. Convergence check (L-infinity norm of the update)
    err = max(abs(T(:) - Told(:)));
    res_history(end+1) = err;
    if err < tol
        fprintf('Converged in %d iterations. Final Res: %.2e\n', it,err);
        break;
    end
end
Q1timer=toc;
fprintf('Total time: %.4f seconds\n', Q1timer);

% --- 5. Visualization ---
Plot_Results(T, xc, yc,res_history)

end

% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

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

function Plot_Results(T_vec, x, y,res_history) 
    figure(1);
    hold on
    contourf(x, y, T_vec, 40, 'LineColor','none');
    colorbar; colormap jet; axis equal;
    title('Q1 Temperature (X-Sweep GS)');
    xlabel('x'); ylabel('y');
    % Plot material interface markers
    plot([0 0],[-0.5 0.5],'k--','LineWidth',1);
    plot([-0.5 0.5],[0 0],'k--','LineWidth',1);
    hold off;

    % Plot convergence history
    figure(11);
    semilogy(res_history, '-o', 'LineWidth', 1.5);
    grid on;
    title('Convergence History Q1');
    xlabel('Iteration'); ylabel('Max |T - Told|');
end