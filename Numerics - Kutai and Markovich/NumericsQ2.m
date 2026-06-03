function NumericsQ2()
% =========================================================================
% Project: Advanced Numerical Methods - Question 2
% Question 2 - Steady State Heat Equation with Multigrid
% Method: Geometric Multigrid (V-Cycle) with Line-GS Smoother
% Discretization: Finite Volume Method (Cell-Centered)
% =========================================================================
clc; clear ALL; close all;
tic;

% --- 1. Parameters ---
N = 200;            % Grid size (200x200)
L = 1.0;            % Domain length
h = L/N;            % Mesh size
k1 = 1e-3;          % Low-conductivity material
k2 = 100.0;         % High conductivity (top-right)
tol = 1e-6;         % Convergence tolerance
maxCycles = 50;     % Maximum multigrid cycles
coarseLimit = 25;   % Coarsest grid size threshold
coarseIters = 50;   % Smoothing iterations on coarsest grid

% Profile function for Dirichlet boundaries
bc_func = @(y) 1 - 4 * y.^2;

% --- 2. Grid & Initialization ---
% Creating cell-centered grid using L
x = linspace(-L/2 + h/2, L/2 - h/2, N);
[xc, yc] = meshgrid(x, x);

% Initial guess: Linear interpolation (consistent with Q1 style)
T = zeros(N, N);
bcL = bc_func(yc(:,1)); 
bcR = bc_func(yc(:,end));

for i = 1:N
    % Calculate normalized x-position (from 0 to 1)
    weight = (i-1)/(N-1);
    % Linearly blend left (bcL) and right (bcR) boundary temperatures
    T(:,i) = (1-weight)*bcL + weight*bcR;
end

f = zeros(N, N);    % Source term (RHS)
res_history = [];   % Storage for convergence history
fprintf('Q2: Starting MG V-Cycle (N=%d)\n', N);

% Pre-compute fine grid physics for the residual check in the main loop
[K_fine, Kx_fine, Ky_fine, bc_fine] = Get_Physics(N, h, k1, k2, L, bc_func);

% --- 3. Main Multigrid Loop (V-Cycle) ---
for cycle = 1:maxCycles
    % Execute one V-Cycle (Recursive)
    % isCorr = false: solving for Temperature on the finest level
    T = V_Cycle(T, f, h, k1, k2, false, L, bc_func, coarseLimit, coarseIters);

    % Convergence check: Calculate residual norm on fine grid
    r = Compute_Residual(T, f, h, K_fine, Kx_fine, Ky_fine, bc_fine);
    err = norm(r(:), 2);
    res_history(end+1) = err;
    
    if err < tol
        fprintf('Converged in %d cycles. Final Res: %.2e\n', cycle, err);
        break;
    end
end
Q2timer = toc;
fprintf('Total time: %.4f seconds\n', Q2timer);

% --- 4. Visualization ---
Plot_Results(T, xc, yc,res_history)

end

% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function T = V_Cycle(T, f, h, k1, k2, isCorr, L, bc_func, coarseLimit, coarseIters)
    % V-Cycle Algorithm: Pre-smoothing, Restriction, Coarse solve, Prolongation, Post-smoothing
    N = size(T, 1);
    
    % Fetch physics matrices exactly once per level
    [K, Kx, Ky, bc] = Get_Physics(N, h, k1, k2, L, bc_func);
    if isCorr
        bc = zeros(size(bc)); % Homogeneous BCs for error equation
    end
     
    % Base Case: Solve directly/intense smoothing on the coarsest level
    if N <= coarseLimit
        for i=1:coarseIters
            T = GS_Smoother(T, f, h, K, Kx, Ky, bc);
        end
        return;
    end

    % 1. Pre-Smoothing
    T = GS_Smoother(T, f, h, K, Kx, Ky, bc);

    % 2. Residual Computation & Restriction
    r = Compute_Residual(T, f, h, K, Kx, Ky, bc);
    rc = Restrict(r);

    % 3. Recursive Call for Coarse Grid Error Equation (Ae = r)
    ec = V_Cycle(zeros(size(rc)), rc, 2*h, k1, k2, true, L, bc_func, coarseLimit, coarseIters);
    
    % 4. Prolongation & Correction
    T = T + Prolong(ec);
    
    % 5. Post-Smoothing
    T = GS_Smoother(T, f, h, K, Kx, Ky, bc);
end

function T = GS_Smoother(T, f, h, K, Kx, Ky, bc)
    % GS By-Lines: One sweep in X, followed by one sweep in Y
    T = Sweep_X(T, f, h, K, Kx, Ky, bc);
    T = Sweep_Y(T, f, h, K, Kx, Ky, bc);
end

function T = Sweep_X(T, f, h, K, Kx, Ky, bc)
    % Solve vertical columns implicitly using Thomas Algorithm
    N = size(T, 1);
    for i = 1:N
        % Horizontal coefficients (explicit part)
        if i == 1
            kw = 2*K(:,i)/h^2; ke = Kx(:,i); Tw = bc; Te = T(:,i+1);
        elseif i == N
            kw = Kx(:,i-1); ke = 2*K(:,i)/h^2; Tw = T(:,i-1); Te = bc;
        else
            kw = Kx(:,i-1); ke = Kx(:,i); Tw = T(:,i-1); Te = T(:,i+1);
        end
        ks = [0; Ky(:,i)]; kn = [Ky(:,i); 0]; % Vertical interfaces (Adiabatic ends)
        
        % Construct tridiagonal system
        b = -(kw + ke + ks + kn);
        a = ks; c = kn;
        d = -(kw.*Tw + ke.*Te + f(:,i));
        
        T(:,i) = thomas_solver(a, b, c, d);
    end
end

function T = Sweep_Y(T, f, h, K, Kx, Ky, bc)
    % Solve horizontal rows implicitly using Thomas Algorithm
    N = size(T, 1);
    for j = 1:N
        % Vertical neighbors (explicit part)
        if j == 1, ks = zeros(N,1); Ts = zeros(N,1); else, ks = Ky(j-1,:)'; Ts = T(j-1,:)'; end
        if j == N, kn = zeros(N,1); Tn = zeros(N,1); else, kn = Ky(j,:)'; Tn = T(j+1,:)'; end
        
        % Horizontal neighbors (implicit part)
        kw = [2*K(j,1)/h^2; Kx(j,:)'];
        ke = [Kx(j,:)'; 2*K(j,N)/h^2];
        
        b = -(kw + ke + ks + kn);
        a = kw; c = ke;
        d = -(ks.*Ts + kn.*Tn + f(j,:)');
        
        % Dirichlet contributions to RHS
        d(1) = d(1) - kw(1)*bc(j); d(end) = d(end) - ke(end)*bc(j);
        
        T(j,:) = thomas_solver(a, b, c, d)';
    end
end

function r = Compute_Residual(T, f, h, K, Kx, Ky, bc)
    % Vectorized flux balance computation (r = f - AT)
    N = size(T, 1); 
  
    % Internal fluxes
    FluxE = Kx .* (T(:,2:end) - T(:,1:end-1));
    FluxN = Ky .* (T(2:end,:) - T(1:end-1,:));
    % Boundary fluxes
    FluxW_b = (2*K(:,1)/h^2) .* (T(:,1) - bc);
    FluxE_b = (2*K(:,end)/h^2) .* (bc - T(:,end));
    
    % Net balance per cell
    NetFlux = zeros(N, N);
    NetFlux(:, 2:end-1) = FluxE(:, 2:end) - FluxE(:, 1:end-1);
    NetFlux(:, 1) = FluxE(:,1) - FluxW_b;
    NetFlux(:, end) = FluxE_b - FluxE(:,end);
    NetFlux(2:end-1, :) = NetFlux(2:end-1, :) + (FluxN(2:end, :) - FluxN(1:end-1, :));
    NetFlux(1, :) = NetFlux(1, :) + FluxN(1, :); % Adiabatic South
    NetFlux(end, :) = NetFlux(end, :) - FluxN(end, :); % Adiabatic North
    
    % Compute residual r = f - AT.
    r = f + NetFlux;
    % Note: NetFlux is calculated as the sum of (T_neighbor - T_cell), which approximates
    % the positive Laplacian (grad^2 T). Since the course defines the operator A
    % as the negative Laplacian (-grad^2 T = f), the residual is derived as:
    % r = f - AT = f - (-NetFlux) = f + NetFlux.
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

function [K, Kx, Ky, bc] = Get_Physics(N, h, k1, k2, L, bc_func)
    % Map material properties and precompute interface conductivities
    v = linspace(-L/2 + h/2, L/2 - h/2, N);
    [xc, yc] = meshgrid(v, v);
    K = k1 * ones(N, N);
    K(xc >= 0 & yc >= 0) = k2; % High conductivity quadrant
    
    harm = @(a,b) 2*a.*b./(a+b+eps);
    Kx = harm(K(:, 1:end-1), K(:, 2:end)) / h^2;
    Ky = harm(K(1:end-1, :), K(2:end, :)) / h^2;
    bc = bc_func(v');
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

function Plot_Results(T_vec, x, y,res_history) 
    % Heat map distribution
    figure(2);
    hold on;
    contourf(x, y, T_vec, 40, 'LineColor','none');
    colorbar; colormap jet; axis equal;
    title('Q2 Temperature (Multigrid)');
    xlabel('x'); ylabel('y');
    % Material interface markers
    plot([0 0],[-0.5 0.5],'k--','LineWidth',1);
    plot([-0.5 0.5],[0 0],'k--','LineWidth',1);
    hold off;

    % Multigrid convergence history
    figure(22);
    semilogy(res_history, '-o', 'LineWidth', 1.5);
    grid on;
    title('Convergence History Q2 (V-Cycle)');
    xlabel('Cycle'); ylabel('||r||_2');
end