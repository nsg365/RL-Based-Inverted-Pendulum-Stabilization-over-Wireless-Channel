clc; clear; close all;
load trainedAgent
p = furuta_params();

x = [0; 0; deg2rad(20); 0];

figure('Color','k')
axis equal
axis([-1.5 1.5 -1.5 1.5 -1.5 1.5])
grid on
view(40,30)
hold on

for k = 1:800

    u = getAction(agent,x);
    u = double(u{1});

    % RK4 (same as training)
    x_phys = x;
    x_phys(3) = x(3) + pi;

    k1 = furuta_step(x_phys, u, p);
    k2 = furuta_step(x_phys + 0.5*p.dt*k1, u, p);
    k3 = furuta_step(x_phys + 0.5*p.dt*k2, u, p);
    k4 = furuta_step(x_phys + p.dt*k3, u, p);

    x_phys = x_phys + (p.dt/6)*(k1 + 2*k2 + 2*k3 + k4);

    x = x_phys;
    x(3) = wrapToPi(x(3) - pi);

    alpha = x(1);
    theta = x(3);

    xa = p.L*cos(alpha);
    ya = p.L*sin(alpha);

    xp = xa - p.l*sin(theta)*sin(alpha);
    yp = ya + p.l*sin(theta)*cos(alpha);
    zp = p.l*cos(theta);

    cla
    plot3([0 xa],[0 ya],[0 0],'k','LineWidth',3)
    plot3([xa xp],[ya yp],[0 zp],'r','LineWidth',3)
    plot3(xp,yp,zp,'ro','MarkerFaceColor','r','MarkerSize',8)
    plot3(0,0,0,'ko','MarkerFaceColor','k','MarkerSize',10)

    drawnow
end

disp("theta_err:");
disp(x(3));
disp("theta_dot:");
disp(x(4));
disp("alpha_dot:");
disp(x(2));
