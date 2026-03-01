clear all; close all;

% Load simulation data
load SIData.mat;

% Simulation parameters
dt = 0.05; T = 900;
n = 2; K = 5;
g = 9.81; qs = 920;
rhos = 1037; rhoc = 1424;
R = 4; R1 = 1; R2 = 3; r = R + (K * R); 
r1 = R1 + (K * R1); r2 = R2 + (K * R2);

% Variable initialization
xhat = yhist(:,1); that = zeros(r,1); 
xbar = yhist(:,1); tbar = zeros(r,1); 
tb{1} = zeros(r1,1); tb{2} = zeros(r2,1);

% Arrays to hold the data
xhatArray = []; thatArray = []; 
xbarArray = []; tbarArray = [];  

% RSR-proposed parameters
tprev = that;
delta = 1; lambda = 0.999; 
rho = 1; g_max = 1; alpha = 1e-2; 
Sigma = inv(delta)*eye(r); Lambda = lambda*eye(n); 
Gprev = g_max*eye(r); Upsilon = inv(Gprev); Uprev = Upsilon;

% RSR-Baldi parameters
db = 1; pb = 1; gb = 1;
Gb{1} = inv(db)*eye(r1); Gb{2} = inv(db)*eye(r2);
Sb{1} = zeros(r1-1); Sb{2} = zeros(r2-1);
Fb{1} = zeros(r1-1,r1); Fb{2} = zeros(r2-1,r2);
Vb{1} = zeros(r1); Vb{2} = zeros(r2);
Sb_prev{1} = Sb{1}; Sb_prev{2} = Sb{2}; 
Fb_prev{1} = Fb{1}; Fb_prev{2} = Fb{2}; 
Vb_prev{1} = Vb{1}; Vb_prev{2} = Vb{2};

% Artificial basis function parameters
if K > 0
    fake_noise1 = randn(R1, K) * 0.05; 
    fake_scale1 = 0.5 + rand(R1, K); 
    fake_noise2 = randn(R2, K) * 0.05;
    fake_scale2 = 0.5 + rand(R2, K);
end

%% Simulation
for i = 1:length(uhist)
    % Store estimation data
    xhatArray = [xhatArray xhat]; thatArray = [thatArray that]; 
    xbarArray = [xbarArray xbar]; tbarArray = [tbarArray tbar]; 

    % Invoke actual data
    d = dhist(:,i); y = yhist(:,i); u = uhist(:,i);
    rho = (qs * rhos^2 + y(1) * rhoc^2) / (qs * rhos + y(1) * rhoc);
    t = [thist(1:R1,i); zeros(r1-R1,1); thist(R1+1:R1+R2,i); zeros(r2-R2,1)];

    % Generate real functions
    phi1_real = [u(1) * sqrt((d(2) - d(1)) * 1e5 / rho)];
    phi2_real = rho * g * [u(2); u(2) * (qs + y(1)); (qs + y(1))^2] * 1e-5;
    Phi{1} = phi1_real;
    Phi{2} = phi2_real;

    % Generate artificial functions
    if K > 0
        for k_idx = 1:K
            f_batch = fake_scale1(:, k_idx) .* tanh(phi1_real + ...
                fake_noise1(:, k_idx));
            Phi{1} = [Phi{1}; f_batch];
        end
        for k_idx = 1:K
            f_batch = fake_scale2(:, k_idx) .* tanh(phi2_real + ...
                fake_noise2(:, k_idx));
            Phi{2} = [Phi{2}; f_batch];
        end
    end

    % Basis functions
    Psi = [Phi{1}' zeros(1,r2); zeros(1,r1) Phi{2}'];
    
    timehat = tic;
    % RSR-Proposed
    fhat = [0; d(1)];
    ytilde = y - xhat; zeta = that - tprev;
    gamma = g_max * exp(-alpha * (ytilde' * Lambda * ytilde)^2);
    H1 = diag([weih(that)]); H2 = diag([weih(zeta)]); 
    Gamma = gamma * H2; 
    Upsilon = inv(rho * H1 + Gamma);
    Lambda = inv(lambda * eye(n) + Psi * Sigma * Psi');
    Sigma = 1/lambda * Sigma - ...
        1/lambda * Sigma * Psi' * Lambda * Psi * Sigma;
    P = Upsilon - Upsilon * inv(Sigma + Upsilon) * Upsilon;
    that = that + P * Psi' * ytilde - ...
        P * (inv(Upsilon) - lambda * inv(Uprev)) * that + ...
        2 * P * (Gamma * that - lambda * Gprev * tprev);
    xhat = fhat + Psi * that;
    Gprev = Gamma; Uprev = Upsilon; tprev = that;
    dthat(i) = toc(timehat);

    timebar = tic;
    % RSR-Baldi
    eb = y - xbar;
    fbar = [0; d(1)];
    for j = 1:n
        for k = 1:length(Phi{j})
            vb{j}(k) = weib(tb{j}(k));
            if k < length(Phi{j})
                rpb{j}(k) = sigb(tb{j}(k) + tb{j}(k+1)) + ...
                        sigb(tb{j}(k) - tb{j}(k+1));
                rmb{j}(k) = sigb(tb{j}(k) + tb{j}(k+1)) - ...
                        sigb(tb{j}(k) - tb{j}(k+1));
                rb{j}(k) = rpb{j}(k) * tb{j}(k) + rmb{j}(k) * tb{j}(k+1);
                sb{j}(k) = weib(rb{j}(k));
            end
        end
        Vb{j} = diag(vb{j});
        if length(Phi{j}) > 1
            Sb{j} = diag(sb{j});
            for k = 1:length(Phi{j})-1
                Fb{j}(k, k)   = rpb{j}(k);
                Fb{j}(k, k+1) = rmb{j}(k);
            end
        else
            Sb{j} = 0;
            Fb{j} = zeros(1, length(Phi{j})); 
        end
        Ub{j} = inv(pb * Vb{j} + gb * Fb{j}' * Sb{j} * Fb{j});
        Gb{j} = Gb{j} - (Gb{j} * Phi{j} * Phi{j}' * Gb{j}) / ...
            (1 + Phi{j}' * Gb{j} * Phi{j});
        Pb{j} = Ub{j} - Ub{j} * inv(Ub{j} + Gb{j}) * Ub{j};
        tb{j} = tb{j} + Pb{j} * Phi{j} * eb(j) - ...
            pb * Pb{j} * (Vb{j} - Vb_prev{j}) * tb{j} - ...
            gb * Pb{j} * (Fb{j}' * Sb{j} * Fb{j} - ...
            Fb_prev{j}' * Sb_prev{j} * Fb_prev{j}) * tb{j};
        Sb_prev{j} = Sb{j}; Fb_prev{j} = Fb{j}; Vb_prev{j} = Vb{j};
        xbar(j) = fbar(j) + Phi{j}' * tb{j};
    end
    tbar = [tb{1}; tb{2}];
    dtbar(i) = toc(timebar);

    % Norm error
    ebar = t - tbar; nbar(i) = norm(ebar); 
    ehat = t - that; nhat(i) = norm(ehat);
end

%% RMSE calculation
rbar = sqrt(mean(nbar.^2));
rhat = sqrt(mean(nhat.^2));
simhat = sum(dthat);
simbar = sum(dtbar);

% Print perfromance
fprintf('-----------------------------------------------------\n');
fprintf('|   Algorithm   | Normalized RMSE | Simulation Time |\n');
fprintf('-----------------------------------------------------\n');
fprintf('| %-13s | %15.7e | %15.7e |\n', 'RSR-Baldi', rbar, simbar);
fprintf('| %-13s | %15.7e | %15.7e |\n', 'RSR-Proposed', rhat, simhat);
fprintf('-----------------------------------------------------\n');

%% Function

% Approximation and reweighting
function y = weih(x)
    e = 0.1;
    y = 1 ./ ((abs(x) + e) .* sqrt(x.^2 + e^2));
end

function y = weib(x)
    eb = 0.1; edb = 1e-7;
    y = 1 / ((abs(x) + eb) * sqrt(x^2 + edb));
end

% Reverse sigmoid function
function y = sigb(x)
    y = 1 / (1 + exp(x));
end