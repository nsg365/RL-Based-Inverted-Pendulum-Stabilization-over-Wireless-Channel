"""Utility constants and helpers."""
import numpy as np
from typing import Tuple

# State indices
THETA = 0          # Pendulum angle
ALPHA = 1          # Arm angle
THETA_DOT = 2      # Pendulum angular velocity
ALPHA_DOT = 3      # Arm angular velocity

# Hardware parameters
class HardwareParams:
    """Physical parameters of the Furuta pendulum."""
    
    # Pendulum
    Mp = 0.027          # pendulum mass (kg)
    Lp = 0.191          # pendulum length (m)
    lp = 0.153          # pendulum com distance (m)
    Jp = 1.10e-4        # pendulum moment of inertia (kg·m²)
    
    # Arm/rotor
    Mr = 0.028          # rotor mass (kg)
    Lr = 0.0826         # rotor length (m)
    Jr = 1.23e-4        # rotor moment of inertia (kg·m²)
    
    # Motor & electrical
    Rm = 3.3            # motor resistance (Ω)
    kt = 0.02797        # torque constant (N·m/A)
    km = 0.02797        # back-EMF constant (V·s/rad)
    
    # Damping
    Dr = 0.005          # rotor friction damping
    Dp = 0.001          # pendulum friction damping
    
    # Environment
    g = 9.81            # gravity (m/s²)
    dt = 0.01           # control dt (s) — 100 Hz
    max_voltage = 12.0  # max motor voltage (V)


class Timing:
    """Track control frequency and timing."""
    
    def __init__(self, control_freq: float = 100.0):
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
