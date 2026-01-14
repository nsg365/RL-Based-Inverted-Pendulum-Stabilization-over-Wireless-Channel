function env = furuta_env()

p = furuta_params();

obsInfo = rlNumericSpec([4 1]);
obsInfo.Name = "x";

actInfo = rlNumericSpec([1 1], ...
    'LowerLimit', -p.umax, ...
    'UpperLimit',  p.umax);
actInfo.Name = "tau";

env = rlFunctionEnv(obsInfo, actInfo, @stepFcn, @resetFcn);

    function [nextObs, reward, done, log] = stepFcn(u, log)

        x = log.State;

        % Shift equilibrium: theta = 0 is upright
        x_phys = x;
        x_phys(3) = x(3) + pi;

        % RK4 integration
        k1 = furuta_step(x_phys, u, p);
        k2 = furuta_step(x_phys + 0.5*p.dt*k1, u, p);
        k3 = furuta_step(x_phys + 0.5*p.dt*k2, u, p);
        k4 = furuta_step(x_phys + p.dt*k3, u, p);

        x_phys = x_phys + (p.dt/6)*(k1 + 2*k2 + 2*k3 + k4);

        % Back to shifted coordinates
        x = x_phys;
        x(3) = wrapToPi(x(3) - pi);

        log.State = x;
        nextObs = x;

        % Reward
        theta   = x(3);
        dtheta  = x(4);
        dalpha  = x(2);

        reward = ...
            - 200 * theta^2 ...
            - 50  * dtheta^2 ...
            - 10  * dalpha^2 ...
            - 0.01 * u^2;

        done = false;
    end

    function [x0, log] = resetFcn()
        x0 = [0; 0; deg2rad(20)*(2*rand-1); 0];
        log.State = x0;
    end
end
