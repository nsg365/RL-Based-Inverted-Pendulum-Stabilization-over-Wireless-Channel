clc; clear; close all;

load trainedAgent
p = furuta_params();

% initial state
x = [0; 0; deg2rad(15); 0];
% x = [0; 0; deg2rad(160); 0];


figure('Color','k')
axis equal
axis([-1.5 1.5 -1.5 1.5 -1.5 1.5])
grid on
view(40,30)
hold on

N = 1500;
t = (0:N-1) * p.dt;

alpha_hist     = zeros(1, N);
alpha_dot_hist = zeros(1, N);
theta_hist     = zeros(1, N);
theta_dot_hist = zeros(1, N);


theta_switch = deg2rad(30);   % switch zon


for k = 1:1500

    theta_err = wrapToPi(x(3));

    if abs(theta_err) > theta_switch
        % swing up
        u = energy_swingup(x, p);
        modeStr = 'SWING-UP';
    else
        % RL
        u = getAction(agent, [x(1); x(2); theta_err; x(4)]);
        u = double(u{1});
        modeStr = 'RL';
    end

    % RK4
    x_phys = x;
    x_phys(3) = x(3) + pi;

    k1 = furuta_step(x_phys, u, p);
    k2 = furuta_step(x_phys + 0.5*p.dt*k1, u, p);
    k3 = furuta_step(x_phys + 0.5*p.dt*k2, u, p);
    k4 = furuta_step(x_phys + p.dt*k3, u, p);

    x_phys = x_phys + (p.dt/6)*(k1 + 2*k2 + 2*k3 + k4);
    x = x_phys;
    x(3) = wrapToPi(x(3) - pi);


    alpha     = x(1);
    alpha_dot = x(2);
    theta     = x(3);
    theta_dot = x(4);

    alpha_hist(k)     = alpha;
    alpha_dot_hist(k) = alpha_dot;
    theta_hist(k)     = theta;
    theta_dot_hist(k) = theta_dot;


    % plotting

    xa = p.L*cos(alpha);
    ya = p.L*sin(alpha);

    xp = xa - p.l*sin(theta)*sin(alpha);
    yp = ya + p.l*sin(theta)*cos(alpha);
    zp = p.l*cos(theta);

    cla

    plot3([0 xa],[0 ya],[0 0],'w','LineWidth',3)
    hold on
    plot3([xa xp],[ya yp],[0 zp],'r','LineWidth',3)
    plot3(xp,yp,zp,'ro','MarkerFaceColor','r','MarkerSize',8)
    plot3(0,0,0,'ko','MarkerFaceColor','k','MarkerSize',10)

    text( ...
    -1.45, 1.45, 1.45, ...
    sprintf( ...
        ['MODE       : %s\n\n', ...
         'alpha      = %+7.2f deg\n', ...
         'alpha_dot  = %+7.2f deg/s\n', ...
         'theta      = %+7.2f deg\n', ...
         'theta_dot  = %+7.2f deg/s'], ...
        modeStr, ...
        rad2deg(alpha), rad2deg(alpha_dot), ...
        rad2deg(theta), rad2deg(theta_dot) ...
    ), ...
    'Color','w', ...
    'FontSize',12, ...
    'FontName','Courier', ...
    'VerticalAlignment','top', ...
    'Interpreter','none' ...
);


    drawnow
end

figure('Color','k')


% POSITION STATES (DEGREES)
subplot(2,1,1)
hold on
grid on

plot(t, rad2deg(alpha_hist), 'LineWidth', 2)
plot(t, rad2deg(theta_hist), 'LineWidth', 2)

ylabel('Angle (deg)')
title('Furuta Pendulum States vs Time')

legend( ...
    'arm angle (deg)', ...
    'pendulum angle (deg)', ...
    'Location','northeast' ...
)

set(gca,'Color','k','XColor','w','YColor','w')


% VELOCITY STATES (DEG/S)
subplot(2,1,2)
hold on
grid on

plot(t, rad2deg(alpha_dot_hist), 'LineWidth', 1.5)
plot(t, rad2deg(theta_dot_hist), 'LineWidth', 1.5)

xlabel('Time (s)')
ylabel('Angular velocity (deg/s)')

legend( ...
    'arm velocity (deg/s)', ...
    'pendulum velocity (deg/s)', ...
    'Location','northeast' ...
)

set(gca,'Color','k','XColor','w','YColor','w')



disp("Final theta (deg):");
disp(rad2deg(x(3)));
disp("Final theta_dot (deg/s):");
disp(rad2deg(x(4)));
disp("Final alpha_dot (deg/s):");
disp(rad2deg(x(2)));

