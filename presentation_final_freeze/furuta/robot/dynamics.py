"""Furuta pendulum dynamics for simulation."""
import copy
import numpy as np
from furuta.utils import HardwareParams, THETA, ALPHA, THETA_DOT, ALPHA_DOT


class FurutaDynamics:
    """Simulate Furuta pendulum dynamics using Lagrangian mechanics.

    State: [theta, alpha, theta_dot, alpha_dot]
    - theta: pendulum angle (upright = 0)
    - alpha: arm angle
    - theta_dot: pendulum angular velocity
    - alpha_dot: arm angular velocity

    Action: motor voltage [-12, +12] V
    """

    def __init__(self, params: HardwareParams = None, randomize_frac: float = 0.0):
        """Initialize dynamics.

        Args:
            params: HardwareParams instance. If None, use defaults.
            randomize_frac: If > 0, randomize params by ±randomize_frac on each call to randomize().
        """
        self.params = params or HardwareParams()
        self._nominal = copy.deepcopy(self.params)   # frozen nominal — always perturb from here
        self.randomize_frac = randomize_frac
        self._precompute_constants()

        if randomize_frac > 0:
            self.randomize()

    def _precompute_constants(self):
        """Precompute frequently-used constants."""
        p = self.params
        self._Mp_lp2 = p.Mp * p.lp ** 2
        self._Jr_Mp_Lr2 = p.Jr + p.Mp * p.Lr ** 2
        self._Mp_Lr_lp = p.Mp * p.Lr * p.lp
        self._Jp_Mp_lp2 = p.Jp + p.Mp * p.lp ** 2
        self._Mp_g_lp = p.Mp * p.g * p.lp
        self._2_Mp_lp2 = 2.0 * p.Mp * p.lp ** 2

    def randomize(self):
        """Randomize physical parameters by ±randomize_frac from NOMINAL values."""
        def perturb(val):
            factor = np.random.uniform(1.0 - self.randomize_frac, 1.0 + self.randomize_frac)
            return val * factor

        # Always perturb from the frozen nominal — no cumulative drift
        self.params.Mp = perturb(self._nominal.Mp)
        self.params.lp = perturb(self._nominal.lp)
        self.params.Lr = perturb(self._nominal.Lr)
        self.params.Jr = perturb(self._nominal.Jr)
        self.params.Jp = perturb(self._nominal.Jp)
        self.params.Dr = perturb(self._nominal.Dr)
        self.params.Dp = perturb(self._nominal.Dp)
        self.params.km = perturb(self._nominal.km)
        self.params.kt = perturb(self._nominal.kt)
        self.params.Rm = perturb(self._nominal.Rm)

        self._precompute_constants()
    
    def step(self, state: np.ndarray, voltage: float, dt: float = None) -> np.ndarray:
        """Integrate dynamics for one timestep.
        
        Args:
            state: [theta, alpha, theta_dot, alpha_dot]
            voltage: motor command [-12, +12] V
            dt: timestep. If None, use params.dt
        
        Returns:
            new_state
        """
        if dt is None:
            dt = self.params.dt
        
        theta, alpha, th_dot, al_dot = state
        
        # Motor torque: τ = kt * (Vm - km * α̇) / Rm
        voltage = np.clip(voltage, -self.params.max_voltage, self.params.max_voltage)
        tau = (self.params.kt * (voltage - self.params.km * al_dot)) / self.params.Rm
        
        # Inertia matrix M
        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        
        M11 = self._Mp_lp2 * (sin_th ** 2) + self._Jr_Mp_Lr2
        M12 = self._Mp_Lr_lp * cos_th
        M22 = self._Jp_Mp_lp2
        
        # Coriolis/centripetal terms C
        C1 = self._2_Mp_lp2 * cos_th * sin_th * al_dot * th_dot - self._Mp_Lr_lp * sin_th * th_dot ** 2
        C2 = -self._Mp_lp2 * cos_th * sin_th * al_dot ** 2
        
        # Gravity G (acts on pendulum only)
        G2 = -self._Mp_g_lp * sin_th
        
        # Determinant of M
        det = M11 * M22 - M12 * M12
        
        # RHS of EOMs
        tau_1 = tau - C1 - self.params.Dr * al_dot
        tau_2 = -C2 - G2 - self.params.Dp * th_dot
        
        # Solve M * q̈ = τ
        al_acc = (M22 * tau_1 - M12 * tau_2) / det
        th_acc = (M11 * tau_2 - M12 * tau_1) / det
        
        # Euler integration
        th_dot_new = th_dot + th_acc * dt
        al_dot_new = al_dot + al_acc * dt
        theta_new = theta + th_dot_new * dt
        alpha_new = alpha + al_dot_new * dt
        
        # Wrap arm angle to [-π, π]
        alpha_new = (alpha_new + np.pi) % (2 * np.pi) - np.pi
        
        return np.array([theta_new, alpha_new, th_dot_new, al_dot_new], dtype=np.float32)
