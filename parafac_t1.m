function [C,B,A,RMSE0,iter] = parafac_t1(Core,Pa,C,B,A,NN,thresh,maxit)
% PARAFAC_T1 - Three mode decomposition using PARAFAC Tucker1 ALS.
%
% I/O:  [C,B,A,RMSE,iter] = parafac_t1(D,Q,C0,B0,A0,nonneg,thresh,maxit);
%       [C,B,A] = parafac_t1(D,Q,C0,B0,A0);
%
%       INPUTS:
%       D:   Data entered as an (jk)D(i), with j-outer loop, k-inner loop.
%            Alternatively, enter the scores matrix of the PCA on data D,
%            with loadings entered in Q.
%       Q:   Rank (q) of the Tucker1 PCA decomposition when entering the 
%            data (q>=r, the model rank).  If scores are entered in D then 
%            the transpose of the loadings must be used here.
%       C0,B0,A0: Initial estimates for kCr, jBr and iAr modes, where 
%            r is the rank of the trilinear decomposition.
%       Optional inputs:
%       nonneg: Number of nonnegative iterations for each mode (A,B,C),
%            if scalar entered, it applies to all modes
%       thresh: Termination RMSE difference between consecutive iterations.
%            Enter 'auto' (default) to use automatic termination criterion.
%       maxit: Termination maximum number of iterations.
%       OUTPUTS:
%       C,B,A: PARAFAC-ALS estimates for kCr, jBr and iAr modes
%       RMSE: square root of sum of squares of residuals for each iteration
%       iter: iteration count
%
% See Van Benthem and Keenan, J. Chemom., V.22(5), 345-354, 2008 for details
%
% Dependencies: FCNNLS.M
%
% Copyright 2008 National Technology & Engineering Solutions of Sandia, 
% LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the 
% U.S. Government retains certain rights in this software.

% Mark H. Van Benthem, Sandia National Laboratories, 10/24/2006
% Revised: 12/04/2006, 11/30/2007, 12/04/2007, 06/07/2009, 01/21/2010,
%          06/21/2010, 04/01/2012, 02/21/2015, 08/28/2018

% check input arguments
narginchk(5,8)
if nargin < 5
    error('Input arguments are missing.')
end
dopca = false; % flag to perform PCA if data entered as Tucker1 model
[rowT,colT] = size(Core);
[rowP,colP] = size(Pa);
if colP==rowP && colP == 1 % if data entered as matricized 3-way array
    q = Pa; % size of core array, may be greater than PARAFAC model rank
    rowP = colT; 
    dopca = true;
elseif colT~=colP % ensure entered Tucker1 model sizes are identical 
    error('The scores and loadings are not conformable')
else
    q = colT; % set size of core array
end
[ra,ca] = size(C);  [rb,cb] = size(B);  [rc,cc] = size(A);
S = [ra rb rc];
q = [min(q,ra) min(q,rb) min(q,rc)]; % ensure appropriate core size
if ca~=cb || ca~=cc % check entered factor ranks are identical
    error('A, B and C have different column sizes.')
elseif (rowT~=(S(1)*S(2))) || (rowP~=S(3))
    error('Array size does not match A, B, C dimensions')
end
r = ca; % set model rank
if nargin < 6 || isempty(NN)% check for nonnegativity constraints
    NN = zeros(1,3);
elseif length(NN) == 1
    NN = repmat(NN,1,3);
elseif length(NN) ~= 3
    error('Specify only one or three nonneg entries')
end
auto = false; % automatic termination criteria, off
if nargin < 7 || ischar(thresh) || isempty(thresh)
    auto = true; % automatic termination criteria, on
end
if nargin < 8  || isempty(maxit) || isempty(maxit) % maximum # iterations
    maxit = 1e4;
end
RMSE0 = zeros(1,maxit);
NNck = any(NN<maxit)&auto; % nonnegative least squares switch
if dopca % here the full data set is supplied (vs. Tucker1 inputs)
    [~,din] = sort(S);
    g = S;
else
    [~,din] = sort(S(1:2));
    g = [S(1:2) q(3)];
end
for ii = 1:length(din) % generate the core array (in memory efficient way)
    if din(ii)==2
        Core = reshape(Core',g(3)*g(1),g(2));
    elseif din(ii)==1
        Core = reshape(Core,g(1),g(2)*g(3))';
    end
    [lhs,rhs] = size(Core);
    xpr = lhs < rhs;
    if xpr
        CCxp = Core*Core';
    else
        CCxp = Core'*Core;
    end
    if ii==1,  SSz = trace(CCxp);  end
    opts.disp = 0;
    [Px,E] = eigs(CCxp,q(din(ii)),'lm',opts); % generate principal factors
    E = diag(E);
    if xpr
        Px = (Core'*Px)*diag(1./sqrt(E));
    end
    if din(ii)==3
        Pa = Px; % C-mode eigenvectors
        Core = Core*Pa;
        g(3) = q(3);
    elseif din(ii)==2
        Pb = Px; % B-mode eigenvectors
        Core = Core*Pb;
        g(2) = q(2);
        Core = reshape(Core,g(3),g(1)*g(2))';
    else
        Pc = Px; % A-mode eigenvectors
        Core = Core*Pc;
        g(1) = q(1);
        Core = reshape(Core',g(1)*g(2),g(3));
    end
end

RMSE = Inf;
CtC = (C'*C);  BtB = (B'*B);
% add xA,xB and xC switches for different NN entries, 20100121 mvb
if NN(1)==0, C = Pc'*C; xA = false; else, xA = true; end 
if NN(2)==0, B = Pb'*B; xB = false; else, xB = true; end
if NN(3)==0, xC = false; else, xC = true; end
% redesign the output display text string 20150221 mvb
nitstr = floor(log10(maxit))+1;
outstr = sprintf('%%%dd, RMSE: %%10.4e\\n',nitstr);
backstr = repmat('\b',1,nitstr+19);
fprintf(['Iteration: ',outstr],0,0);
for iter = 1:maxit
    doNN = NN >= iter;
    RMSEo = RMSE;
    % compute (C O B)tD (Khatri-Rao product times reshaped core)
    % change cross-product formation if different NN entered, 20100121 mvb
    if xA, U1 = C'*Pc; else, U1 = C'; end
    if xB, U2 = B'*Pb; else, U2 = B'; end
    U1 = U1*reshape(Core,q(1),q(2)*q(3));
    UtD = zeros(r,q(3));
    for ii = 1:r
        UtD(ii,:) = U2(ii,:)*reshape(U1(ii,:),q(2),q(3));
    end
    [A,AtA]   = lsest(UtD,Pa,A,CtC,BtB,r,S(3),doNN(3)); % least squares A-mode
    
    % compute (A O C)tD
    % change cross-product formation if different NN entered, 20100121 mvb
    if xC, U3 = Pa'*A; else, U3 = A; end
    UtD = zeros(r,q(2));
    for ii = 1:r
        UtD(ii,:) = reshape(U1(ii,:),q(2),q(3))*U3(:,ii);
    end
    [B,BtB]   = lsest(UtD,Pb,B,AtA,CtC,r,S(2),doNN(2)); % least squares B-mode

    % compute (B O A)tD
    % change cross-product formation if different NN entered, 20100121 mvb
    if xB, U2 = Pb'*B; else, U2 = B; end
    U3 = Core*U3;
    UtD = zeros(r,q(1));
    for ii = 1:r
        UtD(ii,:) = reshape(U3(:,ii),q(1),q(2))*U2(:,ii);
    end
    [C,CtC,E] = lsest(UtD,Pc,C,BtB,AtA,r,S(1),doNN(1)); % least squares C-mode
    
    % Check convergence
    RMSE = sqrt(SSz-E);
    RMSE0(iter) = RMSE;
    if auto && NNck && iter>2 && any(doNN) && RMSE>RMSEo
        NN(doNN) = iter;
    elseif auto && iter>2 && RMSE>RMSEo
        break
    elseif ~auto && abs(RMSEo-RMSE)<thresh || iter>=maxit 
        break
    end
    
    % add xA,xB and xC switches for different NN entries, 20100121 mvb
    if NN(1)==iter&&iter~=maxit, C = Pc'*C; CtC = (C'*C); xA = false; end
    if NN(2)==iter&&iter~=maxit, B = Pb'*B; BtB = (B'*B); xB = false; end
    if NN(3)==iter&&iter~=maxit, A = Pa'*A; xC = false; 
    end
    fprintf(backstr);fprintf(outstr,iter,RMSE); % command window display
end
RMSE0 = RMSE0(1:iter);
if NN(1) < iter, C = Pc*C; end
if NN(2) < iter, B = Pb*B; end
if NN(3) < iter, A = Pa*A; end
fprintf(backstr);fprintf(outstr,iter,RMSE);

function [Z,ZtZ,E] = lsest(UtD,Pz,Z,XtX,YtY,r,iz,NN) % least squares
UtU = XtX.*YtY;
Z = Z';
if NN
    try
        [Z,~,E] = fcnnls(UtU,UtD*Pz',Z,Z>0,zeros(r,iz),0);
    catch me % recover from infeasible nnls solution errors
        if strcmp(me.identifier,'MATLAB:posdef')
            fprintf('\nInfeasible NNLS Solution. Randomizing Factors.\n');
            Z = rand(size(Z));
            E = sum(sum(Z.*(2*(UtD*Pz')-(UtU)*Z)));
            fprintf('Iteration:%26s','Randomizing Factors');
        else
            rethrow(me)
        end
    end
else
    Z = (UtU\UtD);
    if nargout == 3 % computation for the residual term
        E = sum(sum(Z*Pz'.*(2*(UtD*Pz')-(UtU)*Z*Pz')));
    end
end
if nargout == 2 % normalize factors
    Z = sparse(diag(1./sqrt(sum(Z.^2,2)))) * Z;
end
Z = Z';
ZtZ = Z'*Z;

