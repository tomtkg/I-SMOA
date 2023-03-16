function ISMOA
% Iterative supervised multi-objective optimization algorithm

%------------------------------- Reference --------------------------------
% T. Takagi, K. Takadama, and H. Sato, Pareto front upconvert by iterative
% estimation modeling and solution sampling, Proceedings of The 12th
% Edition of International Conference Series on Evolutionary Multi-
% Criterion Optimization, Lecture Notes in Computer Science, 2023, 13970:
% 218–230.
%--------------------------------------------------------------------------

% This algorithm is written by Tomoaki Takagi

    %% Parameter setting
    M = 3;     % Number of objectives
    D = 3;     % Number of decision variables
    H = 2.6e4; % Size of uniformly distributed L1 unit vector set
    FE = 150;  % Number of function evaluations
    
    %% Generate the uniformly distributed L1 unit vector set
    [W,N] = ILDPoint(H,M);
    
    %% Generate estimator
    estimator = @(W,X,Y) sim(newrbe(X',Y'),W')';
%     I = ones(1,M);
%     estimator = @(W,X,Y) predictor(W, ...
%         dacefit(X,Y,'regpoly0','corrgauss',I,1e-3*I,1e3*I));
    
    %% Load solution data
    Dec = load("DTLZ2_M3_D12_ILD.dat");
%     Dec = load("DTLZ2_M3_D12_UDH.dat");
    Dec = Dec(:,1:D);
    Obj = DTLZ2(Dec,M);
    
    %%  Iterative loop
    for j = 1 : FE
        % Generate model values
        Y = vecnorm(Obj,1,2); % L1 norm set
        X = Obj./Y;           % L1 unit vector set
        
        % Single solution sampling
        objs = W.*estimator(W,X,Y);
        [~,index] = max(min(pdist2(objs,Obj),[],2));
        
        % Evaluation
        dec = zeros(1,D);
        for i = 1 : D
            dec(i) = estimator(W(index,:),X,Dec(:,i));
        end
        dec = min(max(dec,0),1);
        
        % Upconvert
        Dec = [Dec;dec];
        Obj = [Obj;DTLZ2(dec,M)];
    end
    SaveImage(Obj);
end
