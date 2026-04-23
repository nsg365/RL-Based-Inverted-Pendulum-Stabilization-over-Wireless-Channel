import argparse
import os
import sys
import time
import struct
import csv
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import get_schedule_fn

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import serial
except ImportError:
    print("ERROR: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)

TEST_LQR_ONLY = False

# Proportional Gains
K1 = float(-4.4681 * 0.25)
K2 = float(-1.6210 * 0.50)
K3 = float(-39.5037 * 0.60)
K4 = float(-5.0402 * 0.65)
K5 = float(-0.4387 * 0.60)  # The 5th state gain (Anticipatory Brake)

#Serial Communication

class HardwareInterface:
    """Communicate with Arduino running Furuta firmware."""

    BAUD_RATE = 500000
    PACKET_SIZE = 16  # 4 floats
    FULL_PACKET_SIZE = 18  # sync word (2) + data (16)
    SYNC_WORD = b'\xcd\xab'

    def __init__(self, port: str):
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = self.BAUD_RATE
        self.ser.timeout = 0  # CRITICAL: Non-blocking mode for zero latency

        # MAC FIX: Force hardware lines low BEFORE opening
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()

        # MAC FIX: Trigger clean reset
        time.sleep(float(0.1))
        self.ser.dtr = True
        self.ser.rts = True
        time.sleep(float(2.0))  # Wait for bootloader
        self.ser.reset_input_buffer()

    def get_sensor_data(self) -> tuple:
        """
        Ultra-low latency read: Bypasses macOS in_waiting bugs.
        Uses non-blocking bulk reads to grab data in a single rapid OS call.
        Blocks perfectly until the Arduino's 10ms hardware timer fires.
        """
        raw_buffer = b''
        while True:
            chunk = self.ser.read(1024)
            if chunk:
                raw_buffer += chunk
                last_sync_idx = raw_buffer.rfind(self.SYNC_WORD)

                if last_sync_idx != -1 and (len(raw_buffer) - last_sync_idx) >= self.FULL_PACKET_SIZE:
                    start = last_sync_idx + 2
                    data = raw_buffer[start : start + self.PACKET_SIZE]
                    try:
                        return struct.unpack('<ffff', data)
                    except:
                        pass # If unpacking fails due to corruption, just wait for next packet
            pass # Zero-latency CPU polling

    def send_voltage(self, voltage: float):
        """Send voltage command to motor and force immediate USB transfer."""
        self.ser.write(struct.pack('f', float(voltage)))
        self.ser.flush() # MAC FIX: Forces OS to send data immediately

    def close(self):
        self.send_voltage(float(0.0))
        self.ser.close()


#Reward & State Processing

def compute_reward(pend_pos: float, arm_pos: float, pend_vel: float, arm_vel: float, voltage: float, prev_voltage: float = 0.0) -> float:
    """Compute reward (optimized for hardware fine-tuning)."""
    r = (float(20.0)  * -(float(pend_pos) ** float(2))   # theta — pendulum angle (most important)
       + float(5.0)   * -(float(pend_vel) ** float(2))   # theta_dot — damping term
       + float(1.0)   * -(float(arm_pos) ** float(2))    # alpha — arm angle (baseline)
       + float(1.0)   * -(float(arm_vel) ** float(2))    # alpha_dot — arm velocity (baseline)
       + float(0.5)   * -(float(voltage) ** float(2))    # control effort
       + float(2.0)   * -((float(voltage) - float(prev_voltage)) ** float(2))  # smoothness penalty
       + float(1.0))                                     # alive bonus

    return float(r)

def apply_deadzone_model(voltage: float, deadzone: float = float(0.5), threshold: float = float(0.5), cap: float = float(12.0)) -> float:
    voltage = float(voltage)
    deadzone = float(deadzone)
    threshold = float(threshold)
    cap = float(cap)
    if voltage > threshold:
        voltage += deadzone
    elif voltage < -threshold:
        voltage -= deadzone
    else:
        voltage = float(0.0)

    return float(np.clip(float(voltage), float(-cap), float(cap)))

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
        float(np.cos(float(pend_pos))), float(np.sin(float(pend_pos))),
        float(np.cos(float(arm_pos))), float(np.sin(float(arm_pos))),
        float(pend_vel) / float(15.0),
        float(arm_vel) / float(25.0),
    ]

    if include_prev_voltage:
        obs.append(float(np.clip(float(last_voltage) / float(max(float(voltage_cap), float(1e-6))), float(-1.0), float(1.0))))

    return np.array(obs, dtype=np.float32)


def override_hyperparams(model, lr: float, sde_sample_freq: int):
    """Force new LR and SDE sample-freq onto a loaded checkpoint.

    SAC.load() restores whatever hyperparams the checkpoint was saved with,
    so changes made in the fresh-model branch below never reach --resume
    runs unless we explicitly overwrite them here.
    """
    model.sde_sample_freq = int(sde_sample_freq)

    # LR lives in three places: the float, the schedule callable, and each
    # optimizer's param_groups. Set all three so every code path agrees.
    model.learning_rate = float(lr)
    model.lr_schedule = get_schedule_fn(float(lr))

    optimizers = [
        getattr(model.actor, 'optimizer', None),
        getattr(model.critic, 'optimizer', None),
        getattr(model, 'ent_coef_optimizer', None),
    ]
    for opt in optimizers:
        if opt is None:
            continue
        for pg in opt.param_groups:
            pg['lr'] = float(lr)


#Main

def main():
    parser = argparse.ArgumentParser(description="Hardware fine-tuning")

    parser.add_argument('--port', default="/dev/cu.usbmodem1401", required=False, help='Serial port (e.g., /dev/cu.usbmodem1401)')
    parser.add_argument('--vcap', type=float, default=float(12.0), help='Voltage cap for Proportional safety net')
    parser.add_argument('--rl-vcap', type=float, default=float(12.0), help='Voltage cap for RL (defines action semantics, keep FIXED across training)')
    parser.add_argument('--rl-voltage-limit', type=float, default=None, help='Hard clip on applied RL voltage (adjustable for curriculum, defaults to --rl-vcap)')
    parser.add_argument('--safe-angle', type=float, default=float(0.40), help='Pendulum safety cutoff (~23°)')
    parser.add_argument('--resume-angle', type=float, default=float(np.radians(4)), help='Angle to resume control (°)')
    parser.add_argument('--rl-angle', type=float, default=float(4.0), help='RL control boundary in degrees (SAC controls within, proportional catches outside)')
    parser.add_argument('--boundary-penalty', type=float, default=float(-50.0), help='One-time reward penalty when SAC lets pendulum escape --rl-angle')
    parser.add_argument('--warmup-steps', type=int, default=50, help='Steps to send 0V while settling')

    parser.add_argument('--total-steps', type=int, default=0)
    parser.add_argument('--gradient-steps', type=int, default=500, help='Updates performed DURING FALLEN STATE')
    parser.add_argument('--online-gradient-steps', type=int, default=1, help='SAC updates during normal hardware control')
    parser.add_argument('--train-every', type=int, default=100, help='Run online SAC updates every N collected transitions')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--buffer-size', type=int, default=200000)
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='LR for fresh models AND the override on --resume')
    parser.add_argument('--sde-sample-freq', type=int, default=8, help='Steps between SDE noise resamples (lower = less correlated)')
    parser.add_argument('--freeze-entropy', type=float, default=None,
                        help='Freeze SAC entropy coefficient at this fixed value (e.g. 0.05). '
                             'Default (None) = SB3 auto-tune. For fine-tuning from a good '
                             'checkpoint, freezing at 0.02-0.05 makes the policy more '
                             'deterministic and learning much faster.')
    parser.add_argument(
        '--include-prev-voltage',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Append previous applied voltage to the RL observation',
    )

    parser.add_argument('--model', default=None)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--log', default='hardware_log.csv')
    parser.add_argument('--save-every', type=int, default=1000)

    args = parser.parse_args()
    if args.include_prev_voltage is None:
        args.include_prev_voltage = True

    if args.rl_voltage_limit is None:
        args.rl_voltage_limit = args.rl_vcap

    args.rl_angle_rad = float(np.radians(float(args.rl_angle)))

    device = torch.device("cpu")

    print("\n" + "="*70)
    print("Furuta Hardware Fine-Tuning — SAC + proportional")
    print("="*70)
    print(f"  Port: {args.port}")
    print(f"  Proportional V-Cap: {args.vcap:.1f} V")
    print(f"  RL V-Cap (fixed): {args.rl_vcap:.1f} V")
    print(f"  RL Voltage Limit: {args.rl_voltage_limit:.1f} V")
    print(f"  RL Angle Boundary: {args.rl_angle:.1f}° ({float(np.degrees(args.rl_angle_rad)):.2f}°)")
    print(f"  Boundary Penalty: {args.boundary_penalty:.1f}")
    print(f"  Learning rate: {args.learning_rate:g}")
    print(f"  SDE sample freq: {args.sde_sample_freq}")
    print(f"  Gradient steps on fall: {args.gradient_steps}")
    print(f"  Online: {args.online_gradient_steps} step(s) every {args.train_every} transitions")
    print(f"  Previous voltage in obs: {'yes' if args.include_prev_voltage else 'no'}")

    os.makedirs('models', exist_ok=True)
    log_file = open(args.log, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['step', 'pend_rad', 'arm_rad', 'pend_vel', 'arm_vel', 'voltage', 'reward', 'late', 'controller'])

    print("\nConnecting to hardware...")
    try:
        hw = HardwareInterface(args.port)
        print("✓ Connected")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return


    print("\nInitializing SAC model...")

    class HardwareShapeEnv(gym.Env):
        def __init__(self, include_prev_voltage: bool):
            obs_dim = 7 if include_prev_voltage else 6
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        def reset(self, **kwargs): return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        def step(self, action): return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}

    _dummy_env = DummyVecEnv([lambda: HardwareShapeEnv(args.include_prev_voltage)])

    if args.resume:
        resume_stem = args.resume.replace('.zip', '')     # strip .zip if present
        model = SAC.load(resume_stem, env=_dummy_env, device=device)

        # FIX: build buffer path from the stripped stem so we always land on
        # "<stem>_buffer.pkl" whether the user passed ".zip" or not.
        buf_path = resume_stem + '_buffer.pkl'
        if os.path.exists(buf_path):
            model.load_replay_buffer(buf_path)
            print(f"✓ Loaded replay buffer: {model.replay_buffer.size()} transitions")
        else:
            print(f"⚠ No buffer file at {buf_path} — starting with empty replay")

        # FIX: the checkpoint restores OLD hyperparams. Force the new ones.
        override_hyperparams(model, lr=args.learning_rate,
                             sde_sample_freq=args.sde_sample_freq)
        print(f"✓ Overrode LR={args.learning_rate:g}, sde_sample_freq={args.sde_sample_freq}")

    elif args.model:
        model = SAC.load(args.model.replace('.zip', ''), env=_dummy_env, device=device)
        override_hyperparams(model, lr=args.learning_rate,
                             sde_sample_freq=args.sde_sample_freq)
        print(f"✓ Loaded model (no buffer), overrode LR and SDE freq")

    else:
        model = SAC(
            "MlpPolicy",
            _dummy_env,
            learning_rate=float(args.learning_rate),
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            use_sde=True,
            sde_sample_freq=args.sde_sample_freq,
            device=device,
            verbose=0,
        )

    from stable_baselines3.common.logger import configure
    model.set_logger(configure(folder=None, format_strings=[]))
    model.batch_size = args.batch_size

    # Optional: freeze the entropy coefficient. SAC's auto-alpha tuning
    # keeps the policy exploratory, which fights you when fine-tuning from
    # a good checkpoint. Freezing at a small fixed value makes the policy
    # near-deterministic so updates stop undoing prior learning.
    if args.freeze_entropy is not None:
        import torch as _torch
        fixed = float(args.freeze_entropy)
        model.ent_coef_optimizer = None   # stop SB3 from updating log_alpha
        model.ent_coef_tensor = _torch.tensor(fixed, device=model.device)
        # Some SB3 versions also read from log_ent_coef — pin it too
        if hasattr(model, 'log_ent_coef') and model.log_ent_coef is not None:
            model.log_ent_coef.data = _torch.log(_torch.tensor(fixed, device=model.device))
            model.log_ent_coef.requires_grad = False
        print(f"✓ Entropy coefficient frozen at {fixed}")

    print("✓ Model ready\n")

    step = 0
    late_count = 0
    ep_reward = float(0.0)
    last_voltage = float(0.0)   # voltage applied in the previous step
    prev_voltage = float(0.0)   # voltage applied two steps ago (for smoothness penalty)
    last_obs = None
    last_action = None
    transition_count = 0
    last_controller_was_sac = False

    # Disable Garbage Collection to prevent loop stuttering
    gc.disable()

    try:
        input("\n  >>> Hold pendulum UPRIGHT and press Enter to start <<<\n")

        for _ in range(args.warmup_steps):
            hw.send_voltage(float(0.0))
            time.sleep(float(0.01))

        t_start = time.perf_counter()

        while args.total_steps == 0 or step < args.total_steps:
            sensor_data = hw.get_sensor_data()
            if sensor_data is None: continue

            t_compute_start = time.perf_counter()
            pend_pos, arm_pos, pend_vel, arm_vel = float(sensor_data[0]), float(sensor_data[1]), float(sensor_data[2]), float(sensor_data[3])
            pend_pos = float(((float(pend_pos) + float(np.pi)) % (float(2.0) * float(np.pi))) - float(np.pi))
            obs = state_to_obs(
                pend_pos,
                arm_pos,
                pend_vel,
                arm_vel,
                last_voltage,
                args.rl_vcap,
                args.include_prev_voltage,
            )
            reward = float(compute_reward(float(pend_pos), float(arm_pos), float(pend_vel), float(arm_vel), float(last_voltage), float(prev_voltage)))

            done = abs(float(pend_pos)) > float(args.rl_angle_rad)
            motor_cutoff = abs(float(pend_pos)) > float(args.safe_angle)

            # Boundary penalty if the pendulum crosses 5 degrees
            if abs(float(pend_pos)) > float(np.radians(5.0)):
                reward += float(args.boundary_penalty)

            ep_reward += reward

            # Store the transition caused by the previous applied voltage.
            # Only store if the action was taken by SAC (do not train on proportional data)
            if last_obs is not None and last_action is not None and last_controller_was_sac and not TEST_LQR_ONLY:
                model.replay_buffer.add(
                    obs=last_obs,
                    next_obs=obs,
                    action=last_action,
                    reward=np.array([reward]),
                    done=np.array([done]),
                    infos=[{}],
                )
                transition_count += 1

                should_train_online = (
                    args.online_gradient_steps > 0
                    and args.train_every > 0
                    and transition_count % args.train_every == 0
                    and model.replay_buffer.size() > args.batch_size
                )
                if should_train_online:
                    model.train(gradient_steps=args.online_gradient_steps, batch_size=args.batch_size)

# 3. Safety cutoff — one life per Arduino reset
            if motor_cutoff:
                hw.send_voltage(0.0)
                print(f"\n{step:>6d} | FALLEN — |θ|={np.degrees(pend_pos):+.1f}° | ep_reward={ep_reward:+.1f}")

                # Train on accumulated buffer
                if not TEST_LQR_ONLY and model.replay_buffer.size() > args.batch_size:
                    buf_sz = model.replay_buffer.size()
                    scaled = min(args.gradient_steps, max(50, (buf_sz * 10) // args.batch_size))
                    print(f"         [Training {scaled} steps on {buf_sz} transitions "
                          f"(capped at {args.gradient_steps}, batch={args.batch_size})...]")
                    t0 = time.perf_counter()
                    model.train(gradient_steps=scaled, batch_size=args.batch_size)
                    print(f"         [done in {time.perf_counter()-t0:.1f}s]")

                # Save checkpoint + buffer
                model.save("models/hardware_ckpt_latest")
                model.save_replay_buffer("models/hardware_ckpt_latest_buffer.pkl")
                print("         [saved: models/hardware_ckpt_latest(.zip, _buffer.pkl)]")


                # --- AUTO-RESUME ON PHYSICAL RESET ---
                print("\n         >>> Pendulum Fallen - Motor Disabled <<<")
                print("         >>> ACTION REQUIRED <<<")
                print("         1. Hold the pendulum perfectly upright")
                print("         2. Press the physical RESET button on the Arduino")
                print("         [Waiting for hardware reset...]")

                while True:
                    try:
                        # Send 0V to keep the motor dead while we wait
                        hw.send_voltage(0.0)
                        
                        # Read data. This will spin silently while the Arduino reboots.
                        sensor_data = hw.get_sensor_data()
                        
                        if sensor_data:
                            p_pos = float(sensor_data[0])
                            p_pos = ((p_pos + np.pi) % (2.0 * np.pi)) - np.pi
                            
                            # When the Arduino finishes rebooting and you are holding it upright,
                            # the encoder resets to 0.0. Once safe, we break the loop and resume.
                            if abs(p_pos) < args.resume_angle:
                                print("         [Hardware Reset & Upright Position Detected! Resuming...]")
                                break
                                
                    except Exception:
                        # Catch MAC USB disconnect on physical reset
                        try: hw.ser.close() 
                        except: pass
                        
                        time.sleep(1.0) # Wait for OS to register disconnect
                        
                        # Spin until the Arduino finishes bootloader and reappears
                        while True:
                            try:
                                hw = HardwareInterface(args.port)
                                break # Successfully reconnected!
                            except Exception:
                                time.sleep(0.5)
                # -------------------------------------


                # 0 V warmup so the velocity IIR filter re-settles
                for _ in range(args.warmup_steps):
                    hw.send_voltage(0.0)
                    time.sleep(0.01)
                
                last_voltage = 0.0
                prev_voltage = 0.0
                last_obs = None
                last_action = None
                last_controller_was_sac = False
                ep_reward = 0.0
                if not TEST_LQR_ONLY and getattr(model, 'use_sde', False):
                    model.policy.reset_noise(1)
                
                continue

            # 5. Controller Selection
            active_controller = "Proportional"

            if TEST_LQR_ONLY:
                #Just proportional
                u = -(float(K1) * float(arm_pos) + float(K2) * float(arm_vel) + float(K3) * float(pend_pos) + float(K4) * float(pend_vel) + float(K5) * float(last_voltage))
                voltage = float(np.clip(float(u), float(-args.vcap), float(args.vcap)))
                action = np.array([float(voltage) / float(args.vcap)])
            else:
                # PHASE 2: Hybrid Safety Envelope
                if abs(float(pend_pos)) < float(args.rl_angle_rad):

                    # SAC Control
                    active_controller = "SAC"
                    action, _ = model.predict(obs, deterministic=False)

                    voltage = float(action[0]) * float(args.rl_vcap) if isinstance(action, np.ndarray) else float(action) * float(args.rl_vcap)
                    voltage = float(np.clip(float(voltage), float(-args.rl_voltage_limit), float(args.rl_voltage_limit)))  # curriculum clip
                    voltage = float(apply_deadzone_model(float(voltage)))
                else:
                    # Proportional
                    active_controller = "Proportional_Catch"
                    u = -(float(K1) * float(arm_pos) + float(K2) * float(arm_vel) + float(K3) * float(pend_pos) + float(K4) * float(pend_vel) + float(K5) * float(last_voltage))

                    voltage = float(np.clip(float(u), float(-args.vcap), float(args.vcap)))
                    voltage = float(apply_deadzone_model(float(voltage)))
                    action = np.array([float(np.clip(float(voltage) / float(args.rl_vcap), float(-1.0), float(1.0)))])

            # Track which controller was active for boundary penalty
            last_controller_was_sac = (active_controller == "SAC")

            # 6. Send voltage
            hw.send_voltage(float(voltage))
            prev_voltage = last_voltage   # shift: t-1 becomes t-2 for next step's smoothness penalty
            last_voltage = float(voltage)
            last_obs = obs

            # Store what the motor ACTUALLY saw (post clip + deadzone),
            # not SAC's raw pre-compensation output.
            applied_norm = float(np.clip(voltage / args.rl_vcap, -1.0, 1.0))
            last_action = np.array([applied_norm], dtype=np.float32)

            # 8. Check Latency Metric
            t_compute_elapsed = time.perf_counter() - t_compute_start
            if t_compute_elapsed > float(0.003):
                late_count += 1

            log_writer.writerow([step, float(pend_pos), float(arm_pos), float(pend_vel), float(arm_vel), float(voltage), float(reward), late_count, active_controller])

            if step % 10 == 0:
                print(f"Step {step:6d} | θ: {float(np.degrees(float(pend_pos))):+6.1f}° | theta_dot: {float(np.degrees(float(pend_vel))):+6.1f}° | alpha: {float(np.degrees(float(arm_pos))):+6.1f}° | alpha_dot: {float(np.degrees(float(arm_vel))):+6.1f}° | V: {float(voltage):+6.2f} | late: {late_count} | ctrl: {active_controller}")

            if step > 0 and step % 200 == 0:
                buf_size = model.replay_buffer.size() if hasattr(model, 'replay_buffer') else 0
                print(f"         [diag] buffer={buf_size} | transitions_stored={transition_count} | ep_reward={ep_reward:+.1f}")

            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                model.save(f"models/hardware_ckpt_{step+1:06d}.zip")

            step += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        hw.close()
        log_file.close()
        gc.enable()
        if not TEST_LQR_ONLY:
            model.save("models/hardware_final")
            model.save("models/hardware_ckpt_latest")
            model.save_replay_buffer("models/hardware_ckpt_latest_buffer.pkl")
            print("✓ Saved final model + buffer")

if __name__ == "__main__":
    main()