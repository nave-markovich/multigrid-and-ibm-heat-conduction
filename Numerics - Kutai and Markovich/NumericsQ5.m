function NumericsQ5()
% =========================================================================
% Project: Advanced Numerical Methods - Question 5
% Task: Question 5 - Unsteady Heat Equation on Body-Fitted Mesh
% Method: Finite Volume Method, Fully Implicit in Time, Direct LU Solver
% Discretization: Body-Fitted Triangular Mesh with Minimum-Correction
%                 Non-Orthogonal Flux Treatment
% =========================================================================
clc; clear ALL; close all; tic;

% --- 1. Generate Yuri's Mesh ---
% The mesh generator provides p, t, triCent, edgeTriL, edgeTriR,
% edgeMid, edgeLen, edge_nL, and the geometry parameters a, b.
MeshGenerator;
close all;

% --- 2. Parameters ---
dt = 1e-3;               % Required time step
k1 = 1e-3;               % Low conductivity region
k2 = 100.0;              % High conductivity region (top-right)
tol_ss = 1e-6;           % Steady-state tolerance
maxTime = 500.0;         % Safety time limit
report_times = [1.0, 5.0];

% Outer fixed-point updates for the deferred non-orthogonal correction
maxCorrIter = 10;
tolCorr = 1e-5;

% --- 3. Basic Geometry and Material Map ---
Ntri = size(t, 1);                     % Number of triangular control volumes
Areas = Compute_Triangle_Areas(p, t);  % Cell areas for the mass term
halfSide = a / 2;                      % Half side length of the square domain

% Assign conductivity per triangle according to the top-right quadrant rule
K = k1 * ones(Ntri, 1);
K(triCent(:,1) >= 0 & triCent(:,2) >= 0) = k2;

% --- 4. Precompute Face Data for Fast Assembly and Correction ---
% All geometric quantities that do not change in time are stored once.
geom = Precompute_Geometry(edgeTriL, edgeTriR, edgeMid, edgeLen, edge_nL, triCent, K, halfSide);

% --- 5. Assemble the Implicit Orthogonal System ---
fprintf('Q5: Assembling sparse matrix...\n');
[A, b_fixed] = Assemble_Orthogonal_System(Ntri, geom);

% M is the diagonal mass matrix coming from the cell areas
M = spdiags(Areas, 0, Ntri, Ntri);
LHS = M - dt * A;

% Direct LU factorization: computed once and reused at every time step
[L_mat, U_mat, P_mat, Q_mat] = lu(LHS);

% --- 6. Initial Condition ---
T = zeros(Ntri, 1);      % Initial temperature field
time = 0.0;
report_idx = 1;
reached_ss = false;

fprintf('Q5: Starting fully implicit simulation...\n');

% --- 7. Time Marching Loop ---
while time < maxTime
    T_old = T;           % Previous time level
    T_new = T;           % Initial guess for the current time level

    % Update the deferred non-orthogonal correction inside the current time step
    for iter = 1:maxCorrIter
        % Build the explicit cross-diffusion correction using the current guess
        b_noc = Compute_NonOrthogonal_Correction(T_new, Areas, geom, Ntri);

        % Fully implicit RHS: old solution + fixed BC source + cross correction
        rhs = Areas .* T_old + dt * (b_fixed + b_noc);

        % Solve the linear system using the precomputed LU factors
        T_inner = Q_mat * (U_mat \ (L_mat \ (P_mat * rhs)));

        % Stop the inner updates once the correction becomes small enough
        if max(abs(T_inner - T_new)) < tolCorr
            T_new = T_inner;
            break;
        end

        % Continue the fixed-point update with the new iterate
        T_new = T_inner;
    end

    % Accept the converged temperature for this time step
    T = T_new;
    time = time + dt;

    % Save requested times for the report figures
    if report_idx <= length(report_times) && abs(time - report_times(report_idx)) < dt/2
        Visualize_Solution(p, t, T, time, report_idx);
        report_idx = report_idx + 1;
    end

    % Stop when the solution no longer changes significantly in time
    if max(abs(T - T_old)) < tol_ss && time > 0.5
        reached_ss = true;
        break;
    end
end

% --- 8. Final Reporting ---
if reached_ss
    fprintf('Steady state reached at t = %.3f seconds\n', time);
else
    fprintf('Time limit reached at t = %.3f seconds\n', time);
end

Visualize_Solution(p, t, T, time, 100);
Q5timer = toc;
fprintf('Total time: %.3f seconds\n', Q5timer);
end

% =========================================================================
% GEOMETRY, ASSEMBLY AND CORRECTION FUNCTIONS
% =========================================================================

function Areas = Compute_Triangle_Areas(p, t)
    % Vectorized triangle area calculation using the shoelace formula.
    x1 = p(t(:,1), 1); y1 = p(t(:,1), 2);
    x2 = p(t(:,2), 1); y2 = p(t(:,2), 2);
    x3 = p(t(:,3), 1); y3 = p(t(:,3), 2);
    Areas = 0.5 * abs(x1 .* (y2 - y3) + x2 .* (y3 - y1) + x3 .* (y1 - y2));
end

function geom = Precompute_Geometry(edgeTriL, edgeTriR, edgeMid, edgeLen, edge_nL, triCent, K, halfSide)
    % Precompute all face-based geometric data required by the solver.
    % The Minimum Correction Method split is S_f = E_f + T_f.
    tol = 1e-4;
    harm = @(a,b) 2*a.*b./(a+b+eps);

    % --- Interior faces ---
    int_idx = (edgeTriR > 0);              % Interior faces have two adjacent cells
    L_int = edgeTriL(int_idx);             % Left owner cell for each interior face
    R_int = edgeTriR(int_idx);             % Right neighbor cell for each interior face
    mid_int = edgeMid(int_idx, :);         % Face midpoints
    len_int = edgeLen(int_idx);            % Face lengths
    n_int = edge_nL(int_idx, :);           % Outward normals with respect to the left cell

    d_int = triCent(R_int,:) - triCent(L_int,:);   % Center-to-center vector
    dist_int = sqrt(sum(d_int.^2, 2));             % Distance between the two centroids
    e_int = d_int ./ dist_int;                     % Unit vector in the centroid-to-centroid direction

    % Face-gradient interpolation weight based on projected face position
    rCf = mid_int - triCent(L_int,:);                 % vector from left centroid to face midpoint
    gF  = sum(rCf .* e_int, 2) ./ dist_int;           % projected fraction along C->F line
    gF  = max(0.0, min(1.0, gF));                     % optional safety clamp

    Sf_int = n_int .* len_int;                     % Geometric face vector S_f
    proj_int = sum(Sf_int .* e_int, 2);           % Projection of S_f on the orthogonal direction
    Ef_int = proj_int .* e_int;                   % Orthogonal part E_f
    Tf_int = Sf_int - Ef_int;                     % Non-orthogonal remainder T_f

    % Harmonic averaging is used across material jumps on interior faces
    k_int = harm(K(L_int), K(R_int));
    coeff_int = k_int .* proj_int ./ dist_int;    % Two-point orthogonal transmissibility

    % --- Boundary faces ---
    bnd_idx = (edgeTriR == 0);                    % Boundary faces have no right neighbor
    owner_bnd = edgeTriL(bnd_idx);                % Owning cell of each boundary face
    mid_bnd = edgeMid(bnd_idx, :);                % Boundary face midpoints
    len_bnd = edgeLen(bnd_idx);                   % Boundary face lengths
    n_bnd = edge_nL(bnd_idx, :);                  % Boundary outward normals

    d_bnd = mid_bnd - triCent(owner_bnd,:);       % Vector from cell center to boundary face midpoint
    dist_bnd = sqrt(sum(d_bnd.^2, 2));            % Cell-center to face-midpoint distance
    e_bnd = d_bnd ./ dist_bnd;                    % Unit direction toward the boundary face

    Sf_bnd = n_bnd .* len_bnd;                    % Boundary face vector S_f
    proj_bnd = sum(Sf_bnd .* e_bnd, 2);          % Orthogonal projection on the boundary direction
    Ef_bnd = proj_bnd .* e_bnd;                  % Orthogonal part for the boundary face
    Tf_bnd = Sf_bnd - Ef_bnd;                    % Tangential/non-orthogonal part for the boundary face

    k_bnd = K(owner_bnd);                         % Boundary face conductivity comes from the owner cell

    % Identify which physical boundary each face belongs to
    is_LR = abs(abs(mid_bnd(:,1)) - halfSide) < tol;
    is_TB = abs(abs(mid_bnd(:,2)) - halfSide) < tol;
    is_hole = ~(is_LR | is_TB);
    is_Dir = is_hole | is_LR;                     % Hole and left/right walls are Dirichlet boundaries

    % Dirichlet values prescribed at the boundary face midpoints
    T_dir = zeros(size(owner_bnd));
    T_dir(is_hole) = 2.0;
    T_dir(is_LR) = 1 - 4 * mid_bnd(is_LR,2).^2;

    % Boundary orthogonal coefficient for Dirichlet faces only
    coeff_bnd = zeros(size(owner_bnd));
    coeff_bnd(is_Dir) = k_bnd(is_Dir) .* proj_bnd(is_Dir) ./ dist_bnd(is_Dir);

    % Store interior-face data in a compact struct
    geom.int.L = L_int;    geom.int.R = R_int;
    geom.int.mid = mid_int;    geom.int.len = len_int;
    geom.int.n = n_int;    geom.int.d = d_int;
    geom.int.dist = dist_int;    geom.int.e = e_int;
    geom.int.Sf = Sf_int;    geom.int.Ef = Ef_int;
    geom.int.Tf = Tf_int;    geom.int.k = k_int;
    geom.int.coeff = coeff_int;

    % Store boundary-face data in a compact struct
    geom.bnd.owner = owner_bnd;    geom.bnd.mid = mid_bnd;
    geom.bnd.len = len_bnd;    geom.bnd.n = n_bnd;
    geom.bnd.d = d_bnd;    geom.bnd.dist = dist_bnd;
    geom.bnd.e = e_bnd;    geom.bnd.Sf = Sf_bnd;
    geom.bnd.Ef = Ef_bnd;    geom.bnd.Tf = Tf_bnd;
    geom.bnd.k = k_bnd;    geom.bnd.is_LR = is_LR;
    geom.bnd.is_TB = is_TB;    geom.bnd.is_hole = is_hole;
    geom.bnd.is_Dir = is_Dir;    geom.bnd.T_dir = T_dir;
    geom.bnd.coeff = coeff_bnd;     geom.int.gF = gF;
end

function [A, b_fixed] = Assemble_Orthogonal_System(Ntri, geom)
    % Assemble the orthogonal two-point part of the diffusion operator.
    % Interior-face contributions are added with equal-and-opposite signs.
    row = [geom.int.L; geom.int.R; geom.int.L; geom.int.R];
    col = [geom.int.L; geom.int.R; geom.int.R; geom.int.L];
    val = [-geom.int.coeff; -geom.int.coeff; geom.int.coeff; geom.int.coeff];

    % Dirichlet boundary faces contribute to the diagonal and to a fixed RHS source
    owner_dir = geom.bnd.owner(geom.bnd.is_Dir);
    coeff_dir = geom.bnd.coeff(geom.bnd.is_Dir);
    T_dir = geom.bnd.T_dir(geom.bnd.is_Dir);

    row = [row; owner_dir];
    col = [col; owner_dir];
    val = [val; -coeff_dir];

    A = sparse(row, col, val, Ntri, Ntri);
    b_fixed = accumarray(owner_dir, coeff_dir .* T_dir, [Ntri, 1]);
end

function b_noc = Compute_NonOrthogonal_Correction(T, Areas, geom, Ntri)
    % Compute the deferred non-orthogonal correction using Green-Gauss gradients.
    % The correction uses the T_f part of the split S_f = E_f + T_f.

    % --- 1. Face temperatures for Green-Gauss gradient reconstruction ---
    % Interior face temperature is taken as the average of the two adjacent cells.
    T_f_int = 0.5 * (T(geom.int.L) + T(geom.int.R));

    % Boundary face temperatures are imposed from the physical BCs.
    T_f_bnd = zeros(size(geom.bnd.owner));
    T_f_bnd(geom.bnd.is_hole) = 2.0;
    T_f_bnd(geom.bnd.is_LR) = 1 - 4 * geom.bnd.mid(geom.bnd.is_LR,2).^2;
    T_f_bnd(geom.bnd.is_TB) = T(geom.bnd.owner(geom.bnd.is_TB));

    % --- 2. Green-Gauss cell gradients ---
    % Convert face temperatures into face flux-like sums T_f * S_f.
    fluxX_int = T_f_int .* geom.int.Sf(:,1);    fluxY_int = T_f_int .* geom.int.Sf(:,2);
    fluxX_bnd = T_f_bnd .* geom.bnd.Sf(:,1);    fluxY_bnd = T_f_bnd .* geom.bnd.Sf(:,2);

    % Accumulate contributions to each control volume.
    gradX_sum = accumarray(geom.int.L, fluxX_int, [Ntri, 1]) ...
              - accumarray(geom.int.R, fluxX_int, [Ntri, 1]) ...
              + accumarray(geom.bnd.owner, fluxX_bnd, [Ntri, 1]);

    gradY_sum = accumarray(geom.int.L, fluxY_int, [Ntri, 1]) ...
              - accumarray(geom.int.R, fluxY_int, [Ntri, 1]) ...
              + accumarray(geom.bnd.owner, fluxY_bnd, [Ntri, 1]);

    % Divide by cell area to obtain the Green-Gauss gradient in each triangle.
    gradT_x = gradX_sum ./ Areas;    gradT_y = gradY_sum ./ Areas;

    % --- 3. Interior face correction ---
    % Weighted interpolation of cell gradients to interior faces
    gF = geom.int.gF;          % weight of right cell
    gC = 1.0 - gF;             % weight of left cell

    gradT_f_x = gC .* gradT_x(geom.int.L) + gF .* gradT_x(geom.int.R);
    gradT_f_y = gC .* gradT_y(geom.int.L) + gF .* gradT_y(geom.int.R);

    % Cross-diffusion flux based on the tangential part T_f of the face vector.
    F_cross_int = -geom.int.k .* (gradT_f_x .* geom.int.Tf(:,1) + ...
                                  gradT_f_y .* geom.int.Tf(:,2));

    % Add equal-and-opposite interior-face correction to the two adjacent cells.
    b_noc = accumarray(geom.int.L, F_cross_int, [Ntri, 1]) ...
          - accumarray(geom.int.R, F_cross_int, [Ntri, 1]);

    % --- 4. Dirichlet boundary correction ---
    if any(geom.bnd.is_Dir)
        owner_dir = geom.bnd.owner(geom.bnd.is_Dir);
        gradT_dir_x = gradT_x(owner_dir);        gradT_dir_y = gradT_y(owner_dir);
        Tf_dir = geom.bnd.Tf(geom.bnd.is_Dir, :);
        k_dir = geom.bnd.k(geom.bnd.is_Dir);

        % Boundary cross-diffusion is computed from the owner-cell gradient.
        F_cross_bnd = -k_dir .* (gradT_dir_x .* Tf_dir(:,1) + ...
                                 gradT_dir_y .* Tf_dir(:,2));

        % Add the explicit correction to the owner cells of Dirichlet faces.
        b_noc = b_noc + accumarray(owner_dir, F_cross_bnd, [Ntri, 1]);
    end
end

function Visualize_Solution(p, t, T, time, fig_id)
    % Plot the temperature field on the triangular mesh.
    figure(500 + fig_id); clf;
    patch('Faces', t, 'Vertices', p, 'FaceVertexCData', T, ...
          'FaceColor', 'flat', 'EdgeColor', 'none');
    axis equal; axis tight;
    colorbar; colormap jet;
    xlabel('x'); ylabel('y');

    % Use a different title for the steady-state figure.
    if fig_id == 100
        title(sprintf('Q5 Steady State Temperature (t = %.3f)', time));
    else
        title(sprintf('Q5 Temperature at t = %.3f)', time));
    end
end
