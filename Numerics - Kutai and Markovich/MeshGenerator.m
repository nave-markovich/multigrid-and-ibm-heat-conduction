
%% FV TRIANGULAR MESH DATASET (PURE SCRIPT, NO FUNCTIONS)
% Figures:
%   Fig 1: Triangle table (first 10): centroid, vertices, neighbors, outward unit normals per local edge
%   Fig 2: Unique edge table (first 10): nodes, adjacent triangles, length, midpoint, outward unit normal for triL
%   Fig 3: Whole mesh + circle overlay

clear; clc; close all;

% -----------------
% Parameters
% -----------------
a = 1.0;     % square side length
b = 0.2;     % hole radius
hmax = 0.03; % mesh size

% -----------------
% Generate mesh (PDE Toolbox)
% -----------------
model = createpde();

R1 = [3 4  -a/2 a/2 a/2 -a/2   -a/2 -a/2 a/2 a/2]'; % rectangle
C1 = [1 0 0 b  0 0 0 0 0 0]';                        % circle (padded)
gd = [R1 C1];
ns = char('R1','C1')';
sf = 'R1-C1';

dl = decsg(gd, sf, ns);
geometryFromEdges(model, dl);

msh = generateMesh(model, 'GeometricOrder','linear', 'Hmax', hmax);

% Minimal mesh data
p = msh.Nodes.';        % Nnodes x 2
t = msh.Elements.';     % Ntri   x 3
Ntri = size(t,1);

% -----------------
% Enforce CCW triangle ordering
% -----------------
for i = 1:Ntri
    v = t(i,:);
    r1 = p(v(1),:); r2 = p(v(2),:); r3 = p(v(3),:);
    area2 = (r2(1)-r1(1))*(r3(2)-r1(2)) - (r2(2)-r1(2))*(r3(1)-r1(1));
    if area2 < 0
        t(i,[2 3]) = t(i,[3 2]);
    end
end

% -----------------
% Triangle centroids
% -----------------
triCent = zeros(Ntri,2);
for i = 1:Ntri
    v = t(i,:);
    triCent(i,:) = (p(v(1),:) + p(v(2),:) + p(v(3),:)) / 3;
end

% -----------------
% Outward unit normals per triangle local edge
% Local edges:
%   12: v1->v2
%   23: v2->v3
%   31: v3->v1
% For CCW triangle, outward unit normal of directed edge (dx,dy) is [dy,-dx]/||edge||
% -----------------
nTri = zeros(Ntri,3,2); % (tri, edge, component x/y)

for i = 1:Ntri
    v1 = t(i,1); v2 = t(i,2); v3 = t(i,3);

    % edge 12
    d = p(v2,:) - p(v1,:);
    L = norm(d);
    if L == 0, nTri(i,1,:) = [0 0]; else, nTri(i,1,:) = [d(2), -d(1)]/L; end

    % edge 23
    d = p(v3,:) - p(v2,:);
    L = norm(d);
    if L == 0, nTri(i,2,:) = [0 0]; else, nTri(i,2,:) = [d(2), -d(1)]/L; end

    % edge 31
    d = p(v1,:) - p(v3,:);
    L = norm(d);
    if L == 0, nTri(i,3,:) = [0 0]; else, nTri(i,3,:) = [d(2), -d(1)]/L; end
end

% -----------------
% Build neighbors and unique edges
% -----------------
nbr = zeros(Ntri,3);  % neighbors across 12,23,31 (0 if boundary)

% Collect all edges (3 per triangle)
Ekey = zeros(3*Ntri,2); % sorted node ids
Etri = zeros(3*Ntri,1); % triangle id
Eloc = zeros(3*Ntri,1); % local edge id (1..3)

idx = 0;
for i = 1:Ntri
    v = t(i,:);
    edges = [v(1) v(2); v(2) v(3); v(3) v(1)];
    for e = 1:3
        idx = idx + 1;
        Ekey(idx,:) = sort(edges(e,:));
        Etri(idx)   = i;
        Eloc(idx)   = e;
    end
end

% Sort edges by key so duplicates are adjacent
[Es, perm] = sortrows(Ekey);
EtriS = Etri(perm);
ElocS = Eloc(perm);

% Unique edge arrays
edgeNodes = zeros(0,2);
edgeTriL  = zeros(0,1);
edgeTriR  = zeros(0,1);
edgeLen   = zeros(0,1);
edgeMid   = zeros(0,2);
edge_nL   = zeros(0,2); % outward unit normal w.r.t triL

k = 1;
Nall = size(Es,1);
while k <= Nall
    if k < Nall && all(Es(k,:) == Es(k+1,:))
        % interior edge shared by two triangles
        nd = Es(k,:);

        tA = EtriS(k);   eA = ElocS(k);
        tB = EtriS(k+1); eB = ElocS(k+1);

        % neighbors
        nbr(tA,eA) = tB;
        nbr(tB,eB) = tA;

        % store edge
        edgeNodes(end+1,:) = nd; %#ok<SAGROW>
        edgeTriL(end+1,1)  = tA; %#ok<SAGROW>
        edgeTriR(end+1,1)  = tB; %#ok<SAGROW>

        r1 = p(nd(1),:); r2 = p(nd(2),:);
        edgeLen(end+1,1) = norm(r2-r1); %#ok<SAGROW>
        edgeMid(end+1,:) = 0.5*(r1+r2); %#ok<SAGROW>

        edge_nL(end+1,:)  = squeeze(nTri(tA,eA,:)).'; %#ok<SAGROW>

        k = k + 2;
    else
        % boundary edge
        nd = Es(k,:);

        tA = EtriS(k); eA = ElocS(k);

        edgeNodes(end+1,:) = nd; %#ok<SAGROW>
        edgeTriL(end+1,1)  = tA; %#ok<SAGROW>
        edgeTriR(end+1,1)  = 0;  %#ok<SAGROW>

        r1 = p(nd(1),:); r2 = p(nd(2),:);
        edgeLen(end+1,1) = norm(r2-r1); %#ok<SAGROW>
        edgeMid(end+1,:) = 0.5*(r1+r2); %#ok<SAGROW>

        edge_nL(end+1,:)  = squeeze(nTri(tA,eA,:)).'; %#ok<SAGROW>

        k = k + 1;
    end
end

Nedge = size(edgeNodes,1);

% -----------------
% Table 1: first 10 triangles
% -----------------
Ktri = min(10,Ntri);
TriTable10 = table( (1:Ktri)', ...
    triCent(1:Ktri,1), triCent(1:Ktri,2), ...
    t(1:Ktri,1), t(1:Ktri,2), t(1:Ktri,3), ...
    nbr(1:Ktri,1), nbr(1:Ktri,2), nbr(1:Ktri,3), ...
    squeeze(nTri(1:Ktri,1,1)), squeeze(nTri(1:Ktri,1,2)), ...
    squeeze(nTri(1:Ktri,2,1)), squeeze(nTri(1:Ktri,2,2)), ...
    squeeze(nTri(1:Ktri,3,1)), squeeze(nTri(1:Ktri,3,2)), ...
    'VariableNames', {'triID','cx','cy','v1','v2','v3', ...
                      'nbr12','nbr23','nbr31', ...
                      'n12x','n12y','n23x','n23y','n31x','n31y'});

figure(1); clf;
set(gcf,'Name','Triangle table (first 10)','NumberTitle','off','Position',[70 90 1250 430]);
uitable('Data', TriTable10{:,:}, ...
        'ColumnName', TriTable10.Properties.VariableNames, ...
        'Units','normalized','Position',[0.02 0.02 0.96 0.92]);

% -----------------
% Table 2: first 10 unique edges
% -----------------
Kedge = min(10,Nedge);
EdgeTable10 = table( (1:Kedge)', ...
    edgeNodes(1:Kedge,1), edgeNodes(1:Kedge,2), ...
    edgeTriL(1:Kedge), edgeTriR(1:Kedge), ...
    edgeLen(1:Kedge), ...
    edgeMid(1:Kedge,1), edgeMid(1:Kedge,2), ...
    edge_nL(1:Kedge,1), edge_nL(1:Kedge,2), ...
    'VariableNames', {'edgeID','n1','n2','triL','triR','len','mx','my','nLx','nLy'});

figure(2); clf;
set(gcf,'Name','Unique edge table (first 10)','NumberTitle','off','Position',[90 130 1100 430]);
uitable('Data', EdgeTable10{:,:}, ...
        'ColumnName', EdgeTable10.Properties.VariableNames, ...
        'Units','normalized','Position',[0.02 0.02 0.96 0.92]);

% -----------------
% Figure 3: whole mesh + circle overlay
% -----------------
figure(3); clf;
set(gcf,'Name','Whole mesh','NumberTitle','off','Position',[130 60 850 750]);
triplot(t, p(:,1), p(:,2), 'LineWidth', 0.5);
axis equal; grid on; hold on;
title('Triangular mesh (with hole circle overlay)');
xlabel('x'); ylabel('y');

th = linspace(0,2*pi,400);
plot(b*cos(th), b*sin(th), 'LineWidth', 2);
hold off


