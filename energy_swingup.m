function tau = energy_swingup(x, p)
% ENERGY-BASED SWING-UP FOR FURUTA PENDULUM
% theta = 0  -> upright
% theta = pi -> downward

theta  = x(3);
dtheta = x(4);

m  = p.m;
l  = p.l;
g  = p.g;
Jp = p.Ip;

% equivalent inertia
% Jbar = 2*Jp + 0.5*m*l^2;
Jbar = Jp + m*l^2;

% Total pendulum energy (zero at upright)
E = 0.5 * Jbar * dtheta^2 + m*g*l*(1 + cos(theta));

% Desired energy (upright)
E_des = 2*m*g*l;
% E_des = 0;

% Energy gain (KEEP SMALL)
kE = 0.3;

% Core energy shaping law
% tau = kE * (E - E_des) * dtheta * cos(theta);
tau = kE * (E - E_des) * sign(dtheta);


% Escape dead-zone at bottom
if abs(dtheta) < 0.05 && abs(wrapToPi(theta)) < deg2rad(5)
    % tau = 0.15 * p.umax * sign(cos(theta));
    tau = 0.15 * p.umax * sign(theta);

end

% Saturation
tau = max(min(tau, p.umax), -p.umax);

end
