clc;
clear;
close all;

m = 0.027;      % pendulum mass (kg)
l = 0.153;     % pendulum COM length (m)
L = 0.0826;      % arm length (m)
Ip = 1.1e-4;    % pendulum inertia (kg m^2)
ma = 0.028;
Ia = 1.23e-4;    % arm inertia (kg m^2)
g  = 9.81;
Rm = 3.3;
Kt = 0.02797;
Km = 0.02797;

A = [[0,                                                                    1,                                                              0, 0];
[0, -(Km*Kt*(m*l^2 + Ip))/(Rm*((m*L^2 + Ia)*(m*l^2 + Ip) - L^2*l^2*m^2)),       -(L*g*l^2*m^2)/((m*L^2 + Ia)*(m*l^2 + Ip) - L^2*l^2*m^2), 0];
[0,                                                                    0,                                                              0, 1];
[0,         (Km*Kt*L*l*m)/(Rm*((m*L^2 + Ia)*(m*l^2 + Ip) - L^2*l^2*m^2)), (g*l*m*(m*L^2 + Ia))/((m*L^2 + Ia)*(m*l^2 + Ip) - L^2*l^2*m^2), 0]];

B =[                                                           0;
(Kt*(m*l^2 + Ip))/(Rm*((m*L^2 + Ia)*(m*l^2 + Ip) - L^2*l^2*m^2));
                                                               0;
      -(Kt*L*l*m)/(Rm*((m*L^2 + Ia)*(m*l^2 + Ip) - L^2*l^2*m^2))];


%% 3. Pole Placement
% First, verify the system is controllable
if rank(ctrb(A,B)) < 4
    error('The system is not controllable!');
end

%Eigenvalues and eigenvectors
disp('Eigenvalues:');
[V,D,W] = eig(A);
disp(D);
disp('Eigenvectors:');
disp(V);

% Set desired closed-loop poles 
% (Example: 4 stable poles at -10, -11, -12, and -13)
P = [-10 -11 -12 -13];

% Calculate gain matrix K using pole placement
K = place(A, B, P);
disp("Pole placement done!")
%Eigenvalues and eigenvectors
disp('Eigenvalues:');
[V,D,W] = eig(A-B*K);
disp(D);
disp('Eigenvectors:');
disp(V);

% Display results
disp('Calculated Gain Matrix K:');
disp(K);

Ts = 0.010;
C = eye(4); % Dummy C matrix for the state-space model
D = zeros(4,1);

sys_c = ss(A, B, C, D);

% Convert to discrete-time using Zero-Order Hold (zoh)
sys_d = c2d(sys_c, Ts, 'zoh');

Ad = sys_d.A;
Bd = sys_d.B;

disp('Discrete A Matrix (Ad):');
disp(Ad);
disp('Discrete B Matrix (Bd):');
disp(Bd);

%% --- Discrete Pole Placement ---
% Verify the discrete system is controllable
if rank(ctrb(Ad, Bd)) < 4
    error('The discrete system is not controllable!');
end

% Set desired CONTINUOUS closed-loop poles 
Pc = [-10, -11, -12, -13];

% Map continuous poles to DISCRETE poles using z = exp(s*Ts)
Pd = exp(Pc * Ts);

disp('Mapped Discrete Target Poles (z-plane):');
disp(Pd);

% Calculate gain matrix Kd using discrete pole placement
Kd = place(Ad, Bd, Pd);

disp('-----------------------------');
disp('Calculated DISCRETE Gain Matrix Kd (Ready for Arduino):');
disp(Kd);

% Verify Discrete Eigenvalues
disp('Closed-Loop Discrete Eigenvalues [eig(Ad - Bd*Kd)]:');
disp(eig(Ad - Bd*Kd)');
