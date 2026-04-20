clc; clear; close all;

p = furuta_params();

% Initial condition (near downward)
x = [0; 0; pi-0.2; 0];

% Dummy LQR gain (replace with yours)
K = [0 0 30 6];

T = 10;
N = T / p.dt;

X = zeros(4, N);

for k = 1:N
    u = furuta_env(x, K, p);
    xdot = furuta_step(x, u, p);
    x = x + p.dt * xdot;
    X(:,k) = x;
end

t = (0:N-1)*p.dt;

figure;
subplot(2,1,1)
plot(t, X(3,:)); grid on;
ylabel('\theta (rad)')

subplot(2,1,2)
plot(t, X(1,:)); grid on;
ylabel('\alpha (rad)')
xlabel('Time (s)')
