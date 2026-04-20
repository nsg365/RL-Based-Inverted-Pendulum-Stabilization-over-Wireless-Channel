clc;
clear;
close all;

m=0.75;L=1;l=1;Ip=1;Ia=1;g=9.81;

    a = Ia + m*L^2;
    b = m*L*l;
    c = Ip + m*l^2;
    d = m*g*l;

    Delta = a*c - b^2;

    A = [ 0  1     0           0;
          0  0  -b*d/Delta    0;
          0  0     0          1;
          0  0   a*d/Delta    0 ];

    B = [ 0;
        c/Delta;
        0;
        -b/Delta ];

[V,D] = eig(A);
V
D

p = [-1+0.5i -1-0.5i -1.3 -1.5];
K = place(A,B,p)

[V1,D1] = eig(A-B*K);
V1
D1

time = [0 8];

x0 = [0; 0; 0.25; 0];
u = 0;

[t,x] = ode45(@(t,x) pendulum(t,x,u,m,L,l,Ip,Ia,g,A,B,K),time,x0);

plot(t,x);
legend('arm angle','arm velocity','pendulum angle','pendulum velocity');

% alpha = x(:,1);
% theta = x(:,3);
% 
% xa = L*cos(alpha);
% ya = L*sin(alpha);
% 
% xp = xa - l*(sin(theta).*sin(alpha));
% yp = ya + l*(sin(theta).*cos(alpha));
% zp = l*cos(theta);
% 
% figure
% for k = 1:5:length(xp)
%     clf
%     hold on
% 
%     % Arm (lies in XY plane)
%     plot3([0 xa(k)], [0 ya(k)], [0 0], 'k', 'LineWidth', 3)
% 
%     % Pendulum
%     plot3([xa(k) xp(k)], [ya(k) yp(k)], [0 zp(k)], 'r', 'LineWidth', 3)
% 
%     % Pendulum tip
%     plot3(xp(k), yp(k), zp(k), 'ro', 'MarkerSize', 8, 'MarkerFaceColor','r')
% 
%     % Base
%     plot3(0,0,0,'ko','MarkerSize',10,'MarkerFaceColor','k')
% 
%     axis equal
%     axis([-1.5 1.5 -1.5 1.5 -1.5 1.5])
%     grid on
%     view(40,30)
% 
%     drawnow
%     pause(0.3);
% end


function xdot = pendulum(t,x,u,m,L,l,Ip,Ia,g,A,B,K)

    xdot = zeros(4,1);
    xdot(1) = x(2); %alpha dot
    xdot(3) = x(4); %theta dot

    tau = -K*x;
    tau = tau(1);

    a = x(1);
    a_dot = x(2);
    th = x(3);
    th_dot = x(4);

    xdot(2) = (m*L*l*(th_dot^2)*sin(th) + tau - m*L*l*cos(th)*((m*(l^2)*(a_dot^2)*sin(th)*cos(th) + m*g*l*sin(th))/(Ip + m*l^2)) - 2*m*(l^2)*a_dot*th_dot*sin(th)*cos(th))*((Ip + m*(l^2))/((Ip + m*(l^2))*(Ia + m*(L^2) + m*(l^2)*(sin(th)^2)) - (m*L*l*cos(th))^2));
    xdot(4) = (m*(l^2)*(a_dot^2)*sin(th)*cos(th) + m*g*l*sin(th) - m*L*l*cos(th)*((m*L*l*(th_dot^2)*sin(th) + tau - 2*m*(l^2)*a_dot*th_dot*sin(th)*cos(th))/(Ia + m*(L^2) + m*(l^2)*(sin(th)^2))))*(Ia + m*(L^2) + m*(l^2)*(sin(th)^2))/((Ip + m*(l^2))*(Ia + m*(L^2) + m*(l^2)*(sin(th)^2)) - (m*L*l*cos(th))^2);
end

