"""Gymnasium environment for Furuta pendulum stabilization."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple

from furuta.utils import HardwareParams, THETA, ALPHA, THETA_DOT, ALPHA_DOT
from furuta.robot.dynamics import FurutaDynamics


def stabilization_reward(state: np.ndarray, action: float) -> float:
    """Reward function for stabilization task.
    
    Minimize pendulum angle, angular velocity, and control effort.
    Large penalty if pendulum falls beyond safety threshold.
    """
    theta = state[THETA]
    th_dot = state[THETA_DOT]
    
    # Large penalty if fallen
    if abs(theta) > 0.35:  # ~20 degrees
        return -50.0
    
    # Otherwise: minimize angle + velocity + control effort
    r_theta = -(theta ** 2)
    r_vel = -(th_dot ** 2)
    r_control = -(action ** 2)
    
    reward = 10.0 * r_theta + 0.1 * r_vel + 0.01 * r_control
    reward += 0.5  # survival bonus
    
    return float(reward)


class FurutaSimEnv(gym.Env):
    """Gymnasium environment for Furuta pendulum stabilization.
    
    Action: Normalized voltage [-1, 1] → scales to ±12 V
    Observation: [cos(θ), sin(θ), cos(α), sin(α), θ̇/15, α̇/25]
    
    Task: Keep pendulum upright (θ ≈ 0) for as long as possible.
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        control_freq: float = 100.0,
        max_steps: int = 2500,
        domain_randomize: bool = False,
        randomize_frac: float = 0.15,
    ):
        """Initialize environment.
        
        Args:
            control_freq: Control loop frequency in Hz.
            max_steps: Maximum steps per episode.
            domain_randomize: If True, randomize dynamics on each reset.
            randomize_frac: Parameter randomization fraction (±randomize_frac).
        """
        super().__init__()
        
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.max_steps = max_steps
        self.domain_randomize = domain_randomize
        
        self.randomize_frac = randomize_frac

        # Dynamics simulator — pass frac so randomize() uses the right scale
        self.dyn = FurutaDynamics(
            params=HardwareParams(),
            randomize_frac=randomize_frac,
        )
        
        # State: [theta, alpha, theta_dot, alpha_dot]
        self.state = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        
        # Action: normalized voltage [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation: [cos(θ), sin(θ), cos(α), sin(α), θ̇/15, α̇/25]
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32
        )
    
    def _get_obs(self) -> np.ndarray:
        """Get observation from state."""
        theta, alpha, th_dot, al_dot = self.state
        max_th_dot = 15.0
        max_al_dot = 25.0
        
        return np.array([
            np.cos(theta),
            np.sin(theta),
            np.cos(alpha),
            np.sin(alpha),
            th_dot / max_th_dot,
            al_dot / max_al_dot,
        ], dtype=np.float32)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to random near-upright state."""
        super().reset(seed=seed)
        
        if self.domain_randomize:
            self.dyn.randomize()
        
        # Initialize near upright position
        if options is not None:
            theta = float(options.get('theta', np.random.uniform(-0.17, 0.17)))
            alpha = float(options.get('alpha', 0.0))
            th_dot = float(options.get('th_dot', np.random.uniform(-0.1, 0.1)))
            al_dot = float(options.get('al_dot', 0.0))
        else:
            theta = float(np.random.uniform(-0.17, 0.17))   # ±~10 degrees
            alpha = 0.0
            th_dot = float(np.random.uniform(-0.1, 0.1))
            al_dot = 0.0
        
        self.state = np.array([theta, alpha, th_dot, al_dot], dtype=np.float32)
        self.step_count = 0
        
        return self._get_obs(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take one environment step.
        
        Args:
            action: Normalized voltage [-1, 1]
        
        Returns:
            obs, reward, terminated, truncated, info
        """
        self.step_count += 1
        
        # Denormalize action to voltage
        voltage = float(action[0]) * self.dyn.params.max_voltage
        
        # Step dynamics
        self.state = self.dyn.step(self.state, voltage, self.dt)
        
        # Compute reward
        reward = stabilization_reward(self.state, float(action[0]))
        
        # Check termination
        theta = self.state[THETA]
        terminated = abs(theta) > 0.35
        truncated = self.step_count >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}
