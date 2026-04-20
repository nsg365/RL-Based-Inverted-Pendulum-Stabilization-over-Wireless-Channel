clc;
clear;
close all;

syms m L l a a_dot th th_dot Ip Ia tau g Kt Km Rm V;

a_sdot = a_dot;

a_ddot = (m*L*l*(th_dot^2)*sin(th) + (Kt*(V - Km*a_dot)/Rm) - m*L*l*cos(th)*((m*(l^2)*(a_dot^2)*sin(th)*cos(th) + m*g*l*sin(th))/(Ip + m*l^2)) - 2*m*(l^2)*a_dot*th_dot*sin(th)*cos(th))*((Ip + m*(l^2))/((Ip + m*(l^2))*(Ia + m*(L^2) + m*(l^2)*(sin(th)^2)) - (m*L*l*cos(th))^2));

th_sdot = th_dot;

th_ddot = (m*(l^2)*(a_dot^2)*sin(th)*cos(th) + m*g*l*sin(th) - m*L*l*cos(th)*((m*L*l*(th_dot^2)*sin(th) + (Kt*(V - Km*a_dot)/Rm) - 2*m*(l^2)*a_dot*th_dot*sin(th)*cos(th))/(Ia + m*(L^2) + m*(l^2)*(sin(th)^2))))*(Ia + m*(L^2) + m*(l^2)*(sin(th)^2))/((Ip + m*(l^2))*(Ia + m*(L^2) + m*(l^2)*(sin(th)^2)) - (m*L*l*cos(th))^2);

f = [a_sdot; a_ddot; th_sdot; th_ddot];

X = [a; a_dot; th; th_dot];

A = jacobian(f,X);
B = jacobian(f,V);

x0 = [0; 0; 0; 0];
V0 = 0;

A_lin = subs(A,[X; V], [x0; V0])

B_lin = subs(B,[X; V], [x0; V0])

f_linearised = subs(f,[X; V],[x0; V0])

% f_linearised = subs(f,[X; tau],[x0; tau0]) + A_lin * [X-x0];

f_linearised = subs(f,[X; V],[x0; V0]) + A_lin * [X-x0] + B_lin * [V - V0]

