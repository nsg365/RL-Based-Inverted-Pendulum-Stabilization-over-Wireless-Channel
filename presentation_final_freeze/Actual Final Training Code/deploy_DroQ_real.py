import argparse
import csv
import gc
import os
import struct
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs("/tmp/fontconfig", exist_ok=True)

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from sb3_contrib import TQC
    TQC_AVAILABLE = True
except ImportError:
    TQC_AVAILABLE = False

try:
    import serial
except ImportError:
    print("ERROR: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)


# Same proportional safety controller used during training.
K1 = float(-4.4681 * 0.25)
K2 = float(-1.6210 * 0.50)
K3 = float(-39.5037 * 0.75)
K4 = float(-5.0402 * 0.60)
K5 = float(-0.4387 * 0.50)


class HardwareInterface:
    """Communicate with Arduino running the Furuta firmware."""

    BAUD_RATE = 500000
    PACKET_SIZE = 16
    FULL_PACKET_SIZE = 18
    SYNC_WORD = b"\xcd\xab"

    def __init__(self, port: str):
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = self.BAUD_RATE
        self.ser.timeout = 0

        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()

        time.sleep(0.1)
        self.ser.dtr = True
        self.ser.rts = True
        time.sleep(2.0)
        self.ser.reset_input_buffer()

    def get_sensor_data(self) -> tuple:
        raw_buffer = b""
        while True:
            chunk = self.ser.read(1024)
            if chunk:
                raw_buffer += chunk
                last_sync_idx = raw_buffer.rfind(self.SYNC_WORD)
                if (
                    last_sync_idx != -1
                    and (len(raw_buffer) - last_sync_idx) >= self.FULL_PACKET_SIZE
                ):
                    start = last_sync_idx + 2
                    data = raw_buffer[start : start + self.PACKET_SIZE]
                    try:
                        return struct.unpack("<ffff", data)
                    except Exception:
                        pass

    def send_voltage(self, voltage: float):
        self.ser.write(struct.pack("f", float(voltage)))
        self.ser.flush()

    def close(self):
        try:
            self.send_voltage(0.0)
        finally:
            self.ser.close()


def stack_obs(frame_deque: deque) -> np.ndarray:
    return np.concatenate(tuple(frame_deque), dtype=np.float32)


def apply_deadzone_model(
    voltage: float,
    deadzone: float = 0.5,
    threshold: float = 0.5,
    cap: float = 12.0,
) -> float:
    voltage = float(voltage)
    if voltage > threshold:
        voltage += deadzone
    elif voltage < -threshold:
        voltage -= deadzone
    else:
        voltage = 0.0
    return float(np.clip(voltage, -cap, cap))


def state_to_obs(
    pend_pos: float,
    arm_pos: float,
    pend_vel: float,
    arm_vel: float,
    last_voltage: float,
    voltage_cap: float,
    include_prev_voltage: bool,
) -> np.ndarray:
    obs = [
        float(np.cos(pend_pos)),
        float(np.sin(pend_pos)),
        float(np.cos(arm_pos)),
        float(np.sin(arm_pos)),
        float(pend_vel) / 15.0,
        float(arm_vel) / 25.0,
    ]
    if include_prev_voltage:
        obs.append(float(np.clip(last_voltage / max(voltage_cap, 1e-6), -1.0, 1.0)))
    return np.array(obs, dtype=np.float32)


def reset_sde_noise_on_device(model, batch_size: int = 1):
    actor = getattr(model, "actor", None)
    if actor is None:
        actor = getattr(getattr(model, "policy", None), "actor", None)
    if actor is None or not getattr(actor, "use_sde", False):
        return

    actor.reset_noise(batch_size=batch_size)

    try:
        actor_device = next(actor.parameters()).device
    except StopIteration:
        actor_device = getattr(model, "device", None)
    action_dist = getattr(actor, "action_dist", None)
    if action_dist is None or actor_device is None:
        return

    for attr in ("exploration_mat", "exploration_matrices"):
        value = getattr(action_dist, attr, None)
        if isinstance(value, torch.Tensor) and value.device != actor_device:
            setattr(action_dist, attr, value.to(actor_device))


class HardwareShapeEnv(gym.Env):
    def __init__(self, include_prev_voltage: bool, frame_stack: int):
        per_frame = 7 if include_prev_voltage else 6
        obs_dim = per_frame * max(1, int(frame_stack))
        self.observation_space = spaces.Box(-1.0, 1.0, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

    def reset(self, **kwargs):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def load_model(args, env, device):
    model_path = str(Path(args.model).with_suffix(""))
    errors = []

    algos = []
    if args.algo in ("auto", "tqc"):
        if TQC_AVAILABLE:
            algos.append(("TQC/DroQ", TQC))
        else:
            errors.append("TQC/DroQ unavailable because sb3_contrib is not installed")
    if args.algo in ("auto", "sac"):
        algos.append(("SAC", SAC))

    for name, cls in algos:
        try:
            model = cls.load(model_path, env=env, device=device)
            reset_sde_noise_on_device(model, batch_size=1)
            return name, model
        except Exception as exc:
            errors.append(f"{name} load failed: {exc}")

    print("ERROR: Could not load model.")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)


def reconnect(port: str) -> HardwareInterface:
    while True:
        try:
            return HardwareInterface(port)
        except Exception:
            time.sleep(0.5)


def wait_for_reset_and_upright(hw: HardwareInterface, args) -> HardwareInterface:
    print("\n         FALLEN - hold upright and press Arduino RESET")
    while True:
        try:
            hw.send_voltage(0.0)
            sensor_data = hw.get_sensor_data()
            if sensor_data:
                p_pos = float(sensor_data[0])
                p_pos = ((p_pos + np.pi) % (2.0 * np.pi)) - np.pi
                if abs(p_pos) < args.resume_angle:
                    print("         Reset and upright detected. Resuming.")
                    return hw
        except Exception:
            try:
                hw.ser.close()
            except Exception:
                pass
            time.sleep(1.0)
            hw = reconnect(args.port)


def main():
    script_dir = Path(__file__).resolve().parent
    default_model = script_dir / "models" / "hardware_ckpt_latest"

    parser = argparse.ArgumentParser(
        description="Furuta Pendulum - deploy trained DroQ/TQC policy"
    )
    parser.add_argument("--port", default="/dev/cu.usbmodem1401")
    parser.add_argument("--model", default=str(default_model))
    parser.add_argument("--algo", choices=["auto", "tqc", "sac"], default="auto")
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="cpu")
    parser.add_argument("--vcap", type=float, default=12.0)
    parser.add_argument("--rl-vcap", type=float, default=12.0)
    parser.add_argument("--rl-voltage-limit", type=float, default=2.0)
    parser.add_argument("--safe-angle", type=float, default=0.40)
    parser.add_argument("--resume-angle", type=float, default=float(np.radians(4)))
    parser.add_argument("--rl-angle", type=float, default=4.0)
    parser.add_argument("--blend-inner-frac", type=float, default=0.7)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--frame-stack", type=int, default=3)
    parser.add_argument("--include-prev-voltage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sde-sample-freq", type=int, default=3)
    parser.add_argument("--lqr-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--log", default="deploy_droq_log.csv")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.frame_stack < 1:
        print("ERROR: --frame-stack must be >= 1")
        return
    args.rl_angle_rad = float(np.radians(args.rl_angle))
    args.blend_inner = args.rl_angle_rad * float(args.blend_inner_frac)

    device = select_device(args.device)

    env = DummyVecEnv([
        lambda: HardwareShapeEnv(args.include_prev_voltage, args.frame_stack)
    ])
    algo_name, model = load_model(args, env, device)

    print("\n" + "=" * 70)
    print("Furuta Pendulum - DEPLOYMENT MODE")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Algo: {algo_name}")
    print(f"  Device: {device}")
    print(f"  RL zone: <= {args.rl_angle:.1f} deg")
    print(f"  Blend zone: {np.degrees(args.blend_inner):.1f} to {args.rl_angle:.1f} deg")
    print(f"  RL voltage limit: +/-{args.rl_voltage_limit:.2f} V")
    print(f"  Frame stack: {args.frame_stack}")
    print(f"  Deterministic policy: {args.deterministic}")
    print(f"  LQR-only smoke test: {args.lqr_only}")

    if args.dry_run:
        print("\nDry run OK: model loaded and observation shape matches.")
        return

    log_file = open(args.log, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "step", "pend_rad", "arm_rad", "pend_vel", "arm_vel",
        "voltage", "late", "controller",
    ])

    print("\nConnecting to hardware...")
    try:
        hw = HardwareInterface(args.port)
        print("Connected")
    except Exception as exc:
        log_file.close()
        print(f"Failed to connect: {exc}")
        return

    step = 0
    late_count = 0
    last_voltage = 0.0
    frame_deque = deque(maxlen=args.frame_stack)

    gc.disable()
    try:
        input("\n  >>> Hold pendulum UPRIGHT and press Enter to start <<<\n")

        for _ in range(args.warmup_steps):
            hw.send_voltage(0.0)
            time.sleep(0.01)

        while args.max_steps == 0 or step < args.max_steps:
            sensor_data = hw.get_sensor_data()
            if sensor_data is None:
                continue

            t_compute_start = time.perf_counter()
            pend_pos, arm_pos, pend_vel, arm_vel = (
                float(sensor_data[0]), float(sensor_data[1]),
                float(sensor_data[2]), float(sensor_data[3]),
            )
            pend_pos = ((pend_pos + np.pi) % (2.0 * np.pi)) - np.pi

            if abs(pend_pos) > args.safe_angle:
                hw.send_voltage(0.0)
                print(f"\n{step:>6d} | FALLEN |theta|={np.degrees(pend_pos):+.1f} deg")
                hw = wait_for_reset_and_upright(hw, args)
                for _ in range(args.warmup_steps):
                    hw.send_voltage(0.0)
                    time.sleep(0.01)
                last_voltage = 0.0
                frame_deque.clear()
                reset_sde_noise_on_device(model, batch_size=1)
                continue

            raw_obs = state_to_obs(
                pend_pos, arm_pos, pend_vel, arm_vel,
                last_voltage, args.rl_vcap, args.include_prev_voltage,
            )
            if len(frame_deque) == 0:
                for _ in range(args.frame_stack):
                    frame_deque.append(raw_obs)
            else:
                frame_deque.append(raw_obs)
            obs = raw_obs if args.frame_stack == 1 else stack_obs(frame_deque)

            prop_u = -(K1 * arm_pos + K2 * arm_vel
                       + K3 * pend_pos + K4 * pend_vel + K5 * last_voltage)
            prop_v = float(np.clip(prop_u, -args.vcap, args.vcap))

            abs_theta = abs(pend_pos)
            if args.lqr_only:
                active_controller = "Proportional"
                voltage = apply_deadzone_model(prop_v, cap=args.vcap)
            elif abs_theta <= args.blend_inner:
                active_controller = algo_name
                if not args.deterministic and step % max(args.sde_sample_freq, 1) == 0:
                    reset_sde_noise_on_device(model, batch_size=1)
                action, _ = model.predict(obs, deterministic=args.deterministic)
                sac_v = float(action[0]) * args.rl_vcap
                sac_v = float(np.clip(sac_v, -args.rl_voltage_limit, args.rl_voltage_limit))
                voltage = apply_deadzone_model(sac_v, cap=args.vcap)
            elif abs_theta < args.rl_angle_rad:
                active_controller = f"{algo_name}_Blend"
                alpha = 1.0 - (abs_theta - args.blend_inner) / (
                    args.rl_angle_rad - args.blend_inner
                )
                if not args.deterministic and step % max(args.sde_sample_freq, 1) == 0:
                    reset_sde_noise_on_device(model, batch_size=1)
                action, _ = model.predict(obs, deterministic=args.deterministic)
                sac_v = float(action[0]) * args.rl_vcap
                sac_v = float(np.clip(sac_v, -args.rl_voltage_limit, args.rl_voltage_limit))
                voltage = apply_deadzone_model(alpha * sac_v + (1.0 - alpha) * prop_v,
                                               cap=args.vcap)
            else:
                active_controller = "Proportional_Catch"
                voltage = apply_deadzone_model(prop_v, cap=args.vcap)

            hw.send_voltage(float(voltage))
            last_voltage = float(voltage)

            t_compute_elapsed = time.perf_counter() - t_compute_start
            if t_compute_elapsed > 0.003:
                late_count += 1

            log_writer.writerow([
                step, pend_pos, arm_pos, pend_vel, arm_vel,
                voltage, late_count, active_controller,
            ])
            if step % 50 == 0:
                log_file.flush()

            if step % max(args.print_every, 1) == 0:
                print(
                    f"Step {step:6d} | theta: {np.degrees(pend_pos):+6.1f} deg | "
                    f"theta_dot: {np.degrees(pend_vel):+6.1f} deg/s | "
                    f"alpha: {np.degrees(arm_pos):+6.1f} deg | "
                    f"alpha_dot: {np.degrees(arm_vel):+6.1f} deg/s | "
                    f"V: {voltage:+6.2f} | late: {late_count} | "
                    f"ctrl: {active_controller}"
                )

            step += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        hw.close()
        log_file.close()
        gc.enable()


if __name__ == "__main__":
    main()
