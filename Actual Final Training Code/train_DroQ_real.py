import argparse
import os
import sys
import time
import struct
import csv
import gc
from collections import deque, namedtuple
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import get_schedule_fn, polyak_update

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import serial
except ImportError:
    print("ERROR: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)

# DroQ via sb3_contrib TQC (10 ensemble critics + high UTD)
try:
    from sb3_contrib import TQC
    DROQ_AVAILABLE = True
except ImportError:
    DROQ_AVAILABLE = False

TEST_LQR_ONLY = False

# ── Prioritized Experience Replay ────────────────────────────────────────────

PrioritizedSamples = namedtuple(
    "PrioritizedSamples", ["samples", "weights", "tree_indices"]
)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional PER with sum-tree. Uniform .sample() is unchanged."""

    def __init__(self, *args, alpha: float = 0.6, eps: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.eps = float(eps)
        cap = 1
        while cap < self.buffer_size:
            cap *= 2
        self._tree_capacity = cap
        self._sum_tree = np.zeros(2 * cap, dtype=np.float64)
        self._min_tree = np.full(2 * cap, np.inf, dtype=np.float64)
        self._max_priority = 1.0

    def _set_priority(self, data_idx: int, priority: float):
        p_alpha = float(priority) ** self.alpha
        tree_idx = data_idx + self._tree_capacity
        self._sum_tree[tree_idx] = p_alpha
        self._min_tree[tree_idx] = p_alpha
        tree_idx //= 2
        while tree_idx >= 1:
            left = 2 * tree_idx
            right = left + 1
            self._sum_tree[tree_idx] = self._sum_tree[left] + self._sum_tree[right]
            self._min_tree[tree_idx] = min(self._min_tree[left], self._min_tree[right])
            tree_idx //= 2

    def _retrieve_leaf(self, target: float) -> int:
        idx = 1
        while idx < self._tree_capacity:
            left = 2 * idx
            if self._sum_tree[left] >= target:
                idx = left
            else:
                target -= self._sum_tree[left]
                idx = left + 1
        return idx - self._tree_capacity

    def add(self, obs, next_obs, action, reward, done, infos):
        slot = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        # New transitions get max priority so they're guaranteed to be seen
        self._set_priority(slot, self._max_priority)

    def sample_prioritized(self, batch_size: int, beta: float, env=None):
        total_p = self._sum_tree[1]
        assert total_p > 0, "All priorities zero in PER buffer"
        segment = total_p / batch_size
        data_indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        size = self.size()
        for i in range(batch_size):
            target = np.random.uniform(segment * i, segment * (i + 1))
            di = self._retrieve_leaf(target)
            tries = 0
            # Guard against sampling past the written region on first few adds
            while di >= size and tries < 10:
                target = np.random.uniform(0.0, total_p)
                di = self._retrieve_leaf(target)
                tries += 1
            data_indices[i] = di
            priorities[i] = self._sum_tree[di + self._tree_capacity]
        probs = priorities / total_p
        weights = (size * probs) ** (-float(beta))
        weights /= weights.max() + 1e-12
        samples = self._get_samples(data_indices, env=env)
        weights_t = torch.as_tensor(
            weights, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        tree_indices = data_indices + self._tree_capacity
        return PrioritizedSamples(samples=samples, weights=weights_t,
                                   tree_indices=tree_indices)

    def update_priorities(self, tree_indices, td_errors):
        td = np.abs(np.asarray(td_errors, dtype=np.float64)) + self.eps
        for leaf_idx, p in zip(tree_indices, td):
            data_idx = int(leaf_idx) - self._tree_capacity
            self._set_priority(data_idx, float(p))
            if p > self._max_priority:
                self._max_priority = float(p)


def per_train_tqc(
    model, gradient_steps: int, batch_size: int,
    beta_start: float = 0.4, beta_end: float = 1.0,
    beta_anneal_steps: int = 50000,
) -> None:
    """
    TQC training step with PER: weighted quantile-huber critic loss +
    proportional priority updates using |Q_pred - Q_target| averaged over
    the 10-critic ensemble and over quantiles.

    Falls back to standard model.train() if the buffer isn't PER (e.g.
    --no-use-per or SAC). This function is ONLY called during the
    fallen-state training window — never online — so it doesn't add
    latency to the 10 ms control loop.
    """
    if not isinstance(model.replay_buffer, PrioritizedReplayBuffer):
        model.train(gradient_steps=gradient_steps, batch_size=batch_size)
        return

    policy = model.policy
    policy.set_training_mode(True)
    optimizers = [model.actor.optimizer, model.critic.optimizer]
    if model.ent_coef_optimizer is not None:
        optimizers.append(model.ent_coef_optimizer)
    model._update_learning_rate(optimizers)

    critic_losses, actor_losses, td_means = [], [], []

    for g in range(gradient_steps):
        # Anneal beta on total gradient steps taken so far
        progress = min(1.0, model._n_updates / max(beta_anneal_steps, 1))
        beta = beta_start + (beta_end - beta_start) * progress

        batch = model.replay_buffer.sample_prioritized(
            batch_size, beta=beta, env=model._vec_normalize_env
        )
        samples = batch.samples
        weights = batch.weights                        # (B, 1)
        obs = samples.observations
        next_obs = samples.next_observations
        actions = samples.actions
        rewards = samples.rewards
        dones = samples.dones

        if model.use_sde:
            model.actor.reset_noise()

        actions_pi, log_prob = model.actor.action_log_prob(obs)
        log_prob = log_prob.reshape(-1, 1)

        ent_coef_loss = None
        if model.ent_coef_optimizer is not None and model.log_ent_coef is not None:
            ent_coef = torch.exp(model.log_ent_coef.detach())
            ent_coef_loss = -(
                model.log_ent_coef * (log_prob + model.target_entropy).detach()
            ).mean()
        else:
            ent_coef = model.ent_coef_tensor

        if ent_coef_loss is not None and model.ent_coef_optimizer is not None:
            model.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            model.ent_coef_optimizer.step()

        with torch.no_grad():
            next_actions, next_log_prob = model.actor.action_log_prob(next_obs)
            next_quantiles = model.critic_target(next_obs, next_actions)
            n_target_q = (model.critic.quantiles_total
                          - model.top_quantiles_to_drop_per_net
                            * model.critic.n_critics)
            next_quantiles, _ = torch.sort(next_quantiles.reshape(batch_size, -1))
            next_quantiles = next_quantiles[:, :n_target_q]
            target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
            target_quantiles = (rewards
                                + (1 - dones) * model.gamma * target_quantiles)
            target_quantiles.unsqueeze_(dim=1)          # (B, 1, n_target_q)

        current_quantiles = model.critic(obs, actions)  # (B, n_critics, n_q)

        # Per-sample quantile Huber loss (replicates SB3's quantile_huber_loss
        # but keeps the batch dim so IS weights can be applied per sample).
        n_q = current_quantiles.shape[-1]
        cum_prob = (torch.arange(n_q, device=current_quantiles.device,
                                 dtype=torch.float32) + 0.5) / n_q
        cum_prob = cum_prob.view(1, 1, -1, 1)
        pairwise = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
        abs_p = torch.abs(pairwise)
        huber = torch.where(abs_p > 1, abs_p - 0.5, pairwise ** 2 * 0.5)
        loss_elt = torch.abs(cum_prob - (pairwise.detach() < 0).float()) * huber
        per_sample_loss = loss_elt.mean(dim=(1, 2, 3))  # (B,)
        critic_loss = (per_sample_loss * weights.squeeze(1)).mean()
        critic_losses.append(critic_loss.item())

        model.critic.optimizer.zero_grad()
        critic_loss.backward()
        model.critic.optimizer.step()

        qf_pi = model.critic(obs, actions_pi).mean(dim=2).mean(dim=1, keepdim=True)
        actor_loss = (ent_coef * log_prob - qf_pi).mean()
        actor_losses.append(actor_loss.item())
        model.actor.optimizer.zero_grad()
        actor_loss.backward()
        model.actor.optimizer.step()

        # TD error for priority update: |Q_pred - Q_target| averaged across
        # critic ensemble and quantiles.
        with torch.no_grad():
            q_pred = current_quantiles.mean(dim=(1, 2))  # (B,)
            q_tgt = target_quantiles.mean(dim=(1, 2))
            td_errors = torch.abs(q_pred - q_tgt).detach().cpu().numpy()
        td_means.append(float(np.mean(td_errors)))
        model.replay_buffer.update_priorities(batch.tree_indices, td_errors)

        if (g % model.target_update_interval) == 0:
            polyak_update(model.critic.parameters(),
                          model.critic_target.parameters(), model.tau)
            polyak_update(model.batch_norm_stats,
                          model.batch_norm_stats_target, 1.0)

        model._n_updates += 1

    if critic_losses:
        print(f"         [PER beta={beta:.3f} critic={np.mean(critic_losses):.4f} "
              f"actor={np.mean(actor_losses):.4f} td={np.mean(td_means):.4f}]")


# ── Frame Stacking ───────────────────────────────────────────────────────────

def stack_obs(frame_deque: deque) -> np.ndarray:
    """O(1)-ish concat of the raw-frame deque into a flat float32 vector."""
    return np.concatenate(tuple(frame_deque), dtype=np.float32)


# Proportional Gains
K1 = float(-4.4681 * 0.25)
K2 = float(-1.6210 * 0.50)
K3 = float(-39.5037 * 0.75)
K4 = float(-5.0402 * 0.60)
K5 = float(-0.4387 * 0.50)  # Anticipatory Brake


# ── Serial Communication ─────────────────────────────────────────────────────

class HardwareInterface:
    """Communicate with Arduino running Furuta firmware."""

    BAUD_RATE        = 500000
    PACKET_SIZE      = 16    # 4 floats
    FULL_PACKET_SIZE = 18    # sync word (2) + data (16)
    SYNC_WORD        = b'\xcd\xab'

    def __init__(self, port: str):
        self.ser          = serial.Serial()
        self.ser.port     = port
        self.ser.baudrate = self.BAUD_RATE
        self.ser.timeout  = 0   # CRITICAL: Non-blocking mode for zero latency

        # MAC FIX: Force hardware lines low BEFORE opening
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()

        # MAC FIX: Trigger clean reset
        time.sleep(0.1)
        self.ser.dtr = True
        self.ser.rts = True
        time.sleep(2.0)  # Wait for bootloader
        self.ser.reset_input_buffer()

    def get_sensor_data(self) -> tuple:
        """
        Ultra-low latency read: Bypasses macOS in_waiting bugs.
        Non-blocking bulk reads. Blocks until Arduino 10ms timer fires.
        """
        raw_buffer = b''
        while True:
            chunk = self.ser.read(1024)
            if chunk:
                raw_buffer   += chunk
                last_sync_idx = raw_buffer.rfind(self.SYNC_WORD)
                if (last_sync_idx != -1
                        and (len(raw_buffer) - last_sync_idx) >= self.FULL_PACKET_SIZE):
                    start = last_sync_idx + 2
                    data  = raw_buffer[start : start + self.PACKET_SIZE]
                    try:
                        return struct.unpack('<ffff', data)
                    except Exception:
                        pass
            pass  # Zero-latency CPU polling

    def send_voltage(self, voltage: float):
        self.ser.write(struct.pack('f', float(voltage)))
        self.ser.flush()  # MAC FIX: Forces OS to send data immediately

    def close(self):
        self.send_voltage(0.0)
        self.ser.close()


# ── Reward & State Processing ────────────────────────────────────────────────

def compute_reward(pend_pos, arm_pos, pend_vel, arm_vel,
                   voltage, prev_voltage=0.0, rl_angle_rad=0.07) -> float:
    # Smooth boundary penalty: quadratic ramp from 50% to 100% of rl_angle
    theta_ratio = abs(float(pend_pos)) / max(rl_angle_rad, 1e-6)
    boundary = -50.0 * max(0.0, (theta_ratio - 0.5) / 0.5) ** 2 if theta_ratio > 0.5 else 0.0
    r = (20.0 * -(float(pend_pos) ** 2)
       +  5.0 * -(float(pend_vel) ** 2)
       +  0.1 * -(float(arm_pos)  ** 2)
       +  1.0 * -(float(arm_vel)  ** 2)
       +  0.5 * -(float(voltage)  ** 2)
       +  2.0 * -((float(voltage) - float(prev_voltage)) ** 2)
       +  1.0
       +  boundary)
    return float(r)


def apply_deadzone_model(voltage, deadzone=0.5, threshold=0.5, cap=12.0):
    voltage = float(voltage)
    if   voltage >  threshold: voltage += deadzone
    elif voltage < -threshold: voltage -= deadzone
    else:                      voltage  = 0.0
    return float(np.clip(voltage, -cap, cap))


def state_to_obs(pend_pos, arm_pos, pend_vel, arm_vel,
                 last_voltage, voltage_cap, include_prev_voltage):
    obs = [
        float(np.cos(pend_pos)), float(np.sin(pend_pos)),
        float(np.cos(arm_pos)),  float(np.sin(arm_pos)),
        float(pend_vel) / 15.0,
        float(arm_vel)  / 25.0,
    ]
    if include_prev_voltage:
        obs.append(float(np.clip(last_voltage / max(voltage_cap, 1e-6), -1.0, 1.0)))
    return np.array(obs, dtype=np.float32)


def override_hyperparams(model, lr: float, sde_sample_freq: int):
    """Force new LR and SDE sample-freq onto a loaded checkpoint."""
    model.sde_sample_freq = int(sde_sample_freq)
    model.learning_rate   = float(lr)
    model.lr_schedule     = get_schedule_fn(float(lr))
    for opt in [getattr(model.actor,  'optimizer', None),
                getattr(model.critic, 'optimizer', None),
                getattr(model, 'ent_coef_optimizer', None)]:
        if opt is None:
            continue
        for pg in opt.param_groups:
            pg['lr'] = float(lr)


def reset_sde_noise_on_device(model, batch_size: int = 1):
    """
    SB3/SB3-contrib gSDE keeps exploration matrices outside the module state.
    After creating/loading a model on MPS, those matrices can remain on CPU
    until reset, which crashes stochastic predict().
    """
    actor = getattr(model, 'actor', None)
    if actor is None:
        actor = getattr(getattr(model, 'policy', None), 'actor', None)
    if actor is None or not getattr(actor, 'use_sde', False):
        return

    actor.reset_noise(batch_size=batch_size)

    try:
        actor_device = next(actor.parameters()).device
    except StopIteration:
        actor_device = getattr(model, 'device', None)
    action_dist = getattr(actor, 'action_dist', None)
    if action_dist is None or actor_device is None:
        return

    for attr in ('exploration_mat', 'exploration_matrices'):
        value = getattr(action_dist, attr, None)
        if isinstance(value, torch.Tensor) and value.device != actor_device:
            setattr(action_dist, attr, value.to(actor_device))


# ── BC Regularization ────────────────────────────────────────────────────────

def proportional_from_obs_tensor(obs: torch.Tensor, rl_vcap: float,
                                 per_frame_dim: int = 7) -> torch.Tensor:
    """Proportional controller action for a batch of observations. Returns (B,1) in [-1,1]."""
    cur = obs[:, -per_frame_dim:]                        # current frame only
    pend_pos = torch.atan2(cur[:, 1], cur[:, 0])         # theta
    arm_pos  = torch.atan2(cur[:, 3], cur[:, 2])         # alpha
    pend_vel = cur[:, 4] * 15.0                          # theta_dot
    arm_vel  = cur[:, 5] * 25.0                          # alpha_dot
    last_v   = (cur[:, 6] * rl_vcap if cur.shape[1] > 6
                else torch.zeros_like(pend_pos))

    u = -(K1 * arm_pos + K2 * arm_vel +
          K3 * pend_pos + K4 * pend_vel + K5 * last_v)
    return torch.clamp(u / rl_vcap, -1.0, 1.0).unsqueeze(1)  # (B, 1)


def run_bc_updates(model, bc_weight: float, n_steps: int,
                   batch_size: int, rl_vcap: float, device,
                   per_frame_dim: int = 7) -> float:
    """
    n_steps BC gradient steps: minimise ||pi(s) - proportional(s)||^2.
    bc_weight decay is managed by the caller.
    Returns mean BC loss (for logging).
    """
    if bc_weight < 1e-6 or model.replay_buffer.size() < batch_size:
        return 0.0

    total_loss = 0.0
    for _ in range(n_steps):
        data         = model.replay_buffer.sample(batch_size, env=model._vec_normalize_env)
        obs          = data.observations.to(device)
        prop_actions = proportional_from_obs_tensor(obs, rl_vcap, per_frame_dim)
        
        # --- FIX: Reset SDE noise to create a fresh computation graph ---
        if getattr(model, 'use_sde', False):
            model.actor.reset_noise(batch_size)
        # ----------------------------------------------------------------
            
        pi_actions, _ = model.actor.action_log_prob(obs)
        bc_loss      = bc_weight * F.mse_loss(pi_actions, prop_actions.detach())
        model.actor.optimizer.zero_grad()
        bc_loss.backward()
        model.actor.optimizer.step()
        total_loss  += bc_loss.item()

    return total_loss / max(n_steps, 1)


# ── Model-Based Bootstrapping (Dyna) ─────────────────────────────────────────

class DynamicsModel(nn.Module):
    """Residual dynamics MLP: (stacked_obs, action) -> delta for current frame."""

    def __init__(self, per_frame_dim: int, frame_stack: int = 1,
                 act_dim: int = 1, hidden: int = 64, device: str = 'cpu'):
        super().__init__()
        self.per_frame_dim = int(per_frame_dim)
        self.frame_stack   = int(frame_stack)
        self.obs_dim       = self.per_frame_dim * self.frame_stack
        self.act_dim       = act_dim
        self._device       = device
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim + act_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),                 nn.SiLU(),
            nn.Linear(hidden, self.per_frame_dim),     # predict current frame delta
        ).to(device)
        self.to(device)
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        # Per-channel bounds matching state_to_obs encoding:
        # [cos(θ), sin(θ), cos(α), sin(α), θ̇/15, α̇/25, (V/Vcap)]
        lo = [-1, -1, -1, -1, -2, -2]
        hi = [ 1,  1,  1,  1,  2,  2]
        if per_frame_dim == 7:
            lo.append(-1); hi.append(1)
        self._obs_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self._obs_hi = torch.tensor(hi, dtype=torch.float32, device=device)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Returns delta for the CURRENT frame only, shape (B, per_frame_dim)."""
        return self.net(torch.cat([obs, act], dim=-1))

    def _shift_stack(self, obs: torch.Tensor, new_current: torch.Tensor) -> torch.Tensor:
        """Shift stack by one and append new_current."""
        if self.frame_stack <= 1:
            return new_current
        return torch.cat([obs[:, self.per_frame_dim:], new_current], dim=-1)

    def fit(self, replay_buffer, batch_size: int = 256,
            n_steps: int = 200) -> float:
        """Fit on real transitions. Returns mean loss over final quarter."""
        if replay_buffer.size() < batch_size:
            return float('nan')

        losses = []
        pfd = self.per_frame_dim
        for k in range(n_steps):
            data     = replay_buffer.sample(batch_size)
            obs      = data.observations.to(self._device)
            act      = data.actions.to(self._device)
            next_obs = data.next_observations.to(self._device)
            # Residual on the CURRENT frame only
            target   = next_obs[:, -pfd:] - obs[:, -pfd:]

            pred = self.forward(obs, act)
            loss = F.mse_loss(pred, target)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if k >= n_steps * 3 // 4:
                losses.append(loss.item())

        return float(np.mean(losses)) if losses else float('nan')

    @torch.no_grad()
    def generate_synthetic_transitions(
        self, replay_buffer, policy, n_seeds: int, horizon: int,
        rl_vcap: float, safe_angle_rad: float,
    ) -> int:
        """Dyna rollouts: sample seeds, roll forward under policy, add to buffer."""
        if replay_buffer.size() < 64:
            return 0

        data  = replay_buffer.sample(n_seeds)
        obs   = data.observations.to(self._device)           # (N, obs_dim)
        alive = torch.ones(n_seeds, 1, device=self._device)
        added = 0
        pfd   = self.per_frame_dim

        for _ in range(horizon):
            act, _ = policy.actor.action_log_prob(obs)
            act    = act.detach()

            delta       = self.forward(obs, act)             # (N, pfd)
            cur_frame   = obs[:, -pfd:]
            new_current = torch.clamp(cur_frame + delta, self._obs_lo, self._obs_hi)
            # Re-normalise trig pairs to unit circle
            for ci, si in [(0, 1), (2, 3)]:
                norm = torch.sqrt(new_current[:, ci]**2 + new_current[:, si]**2).clamp(min=1e-6)
                new_current[:, ci] = new_current[:, ci] / norm
                new_current[:, si] = new_current[:, si] / norm
            next_obs    = self._shift_stack(obs, new_current)

            # Decode CURRENT frame -> raw values for reward
            pend_pos = torch.atan2(cur_frame[:, 1], cur_frame[:, 0]).cpu().numpy()
            arm_pos  = torch.atan2(cur_frame[:, 3], cur_frame[:, 2]).cpu().numpy()
            pend_vel = (cur_frame[:, 4] * 15.0).cpu().numpy()
            arm_vel  = (cur_frame[:, 5] * 25.0).cpu().numpy()
            voltage  = (act[:, 0] * rl_vcap).cpu().numpy()
            prev_v   = ((cur_frame[:, 6] * rl_vcap).cpu().numpy()
                        if cur_frame.shape[1] > 6 else np.zeros(n_seeds))

            rewards = np.array([
                compute_reward(pp, ap, pv, av, v, p2)
                for pp, ap, pv, av, v, p2 in
                zip(pend_pos, arm_pos, pend_vel, arm_vel, voltage, prev_v)
            ])
            fallen   = (np.abs(pend_pos) > safe_angle_rad).astype(np.float32)
            alive_np = alive.squeeze(1).cpu().numpy()

            for i in range(n_seeds):
                if alive_np[i] < 0.5:
                    continue
                replay_buffer.add(
                    obs=obs[i:i+1].cpu().numpy(),
                    next_obs=next_obs[i:i+1].cpu().numpy(),
                    action=act[i:i+1].cpu().numpy(),
                    reward=np.array([rewards[i]]),
                    done=np.array([fallen[i]]),
                    infos=[{}],
                )
                added += 1

            alive = alive * torch.tensor(
                1.0 - fallen, dtype=torch.float32, device=self._device
            ).unsqueeze(1)
            obs = next_obs

        return added


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hardware fine-tuning")

    # ── Original args (unchanged) ─────────────────────────────────────────────
    parser.add_argument('--port', default="/dev/cu.usbmodem1401", required=False,
                        help='Serial port (e.g., /dev/cu.usbmodem1401)')
    parser.add_argument('--vcap',             type=float, default=12.0)
    parser.add_argument('--rl-vcap',          type=float, default=12.0)
    parser.add_argument('--rl-voltage-limit', type=float, default=None)
    parser.add_argument('--safe-angle',       type=float, default=0.40)
    parser.add_argument('--resume-angle',     type=float, default=float(np.radians(4)))
    parser.add_argument('--rl-angle',         type=float, default=4.0,
                        help='RL control boundary in degrees')
    parser.add_argument('--boundary-penalty', type=float, default=-50.0)
    parser.add_argument('--warmup-steps',     type=int,   default=50)
    parser.add_argument('--total-steps',      type=int,   default=0)
    parser.add_argument('--gradient-steps',   type=int,   default=500,
                        help='Updates performed DURING FALLEN STATE')
    parser.add_argument('--online-gradient-steps', type=int, default=1,
                        help='SAC updates during normal hardware control')
    parser.add_argument('--train-every',      type=int,   default=0,
                        help='0 = no online training (recommended for TQC)')
    parser.add_argument('--batch-size',       type=int,   default=256)
    parser.add_argument('--buffer-size',      type=int,   default=200000)
    parser.add_argument('--learning-rate',    type=float, default=3e-4)
    parser.add_argument('--sde-sample-freq',  type=int,   default=3)
    parser.add_argument('--freeze-entropy',   type=float, default=None)
    parser.add_argument('--include-prev-voltage',
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--model',      default=None)
    parser.add_argument('--resume',     default=None)
    parser.add_argument('--log',        default='hardware_log.csv')
    parser.add_argument('--save-every', type=int, default=1000)

    # DroQ / TQC args
    parser.add_argument('--use-droq', action='store_true', default=True)
    parser.add_argument('--utd-ratio', type=int, default=20)
    parser.add_argument('--n-critics', type=int, default=5,
                        help='Number of TQC critic networks (default: 5)')

    # Model-based Dyna args
    parser.add_argument('--use-model-based', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--model-rollout-seeds',   type=int, default=256)
    parser.add_argument('--model-rollout-horizon', type=int, default=3)
    parser.add_argument('--model-train-steps',     type=int, default=300)

    # BC regularization args
    parser.add_argument('--bc-weight', type=float, default=0.0)
    parser.add_argument('--bc-decay',  type=float, default=0.9997)

    # PER args
    parser.add_argument('--use-per', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--per-alpha',             type=float, default=0.6)
    parser.add_argument('--per-beta-start',        type=float, default=0.4)
    parser.add_argument('--per-beta-end',          type=float, default=1.0)
    parser.add_argument('--per-beta-anneal-steps', type=int,   default=50000)

    # Frame stacking
    parser.add_argument('--frame-stack', type=int, default=3)

    args = parser.parse_args()

    # ── Derived / validated args ──────────────────────────────────────────────
    if args.include_prev_voltage is None:
        args.include_prev_voltage = True
    if args.rl_voltage_limit is None:
        args.rl_voltage_limit = args.rl_vcap
    args.rl_angle_rad = float(np.radians(args.rl_angle))
    args.blend_inner  = args.rl_angle_rad * 0.7  # pure SAC zone

    # Frame stack validation — N=1 is a strict no-op
    if args.frame_stack < 1:
        print(f"ERROR: --frame-stack must be >= 1 (got {args.frame_stack})")
        return
    args.per_frame_dim = 7 if args.include_prev_voltage else 6

    # DroQ: UTD ratio overrides the per-step online gradient count
    if args.use_droq:
        if not DROQ_AVAILABLE:
            print("ERROR: --use-droq requires sb3_contrib. Run: pip install sb3_contrib")
            return
        args.online_gradient_steps = args.utd_ratio
        print(f"[DroQ] UTD={args.utd_ratio} "
              f"-> online_gradient_steps overridden to {args.online_gradient_steps}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    algo_str = f"TQC (DroQ), UTD={args.utd_ratio}" if args.use_droq else "SAC"
    print(f"\n{'='*70}\nFuruta Hardware Fine-Tuning\n{'='*70}")
    print(f"  Port: {args.port}  |  V-Cap: {args.vcap}/{args.rl_vcap}  |  Limit: {args.rl_voltage_limit:.1f}V")
    print(f"  RL angle: {args.rl_angle:.1f}°  |  Penalty: {args.boundary_penalty}  |  LR: {args.learning_rate:g}")
    print(f"  Algo: {algo_str}  |  Grad/fall: {args.gradient_steps}  |  Online: {args.online_gradient_steps}/{args.train_every}")
    print(f"  Dyna: {'ON' if args.use_model_based else 'off'}  |  BC: {args.bc_weight:.4f}  |  "
          f"PER: {'ON' if args.use_per else 'off'}  |  Stack: {args.frame_stack}")

    os.makedirs('models', exist_ok=True)
    log_file   = open(args.log, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['step','pend_rad','arm_rad','pend_vel','arm_vel',
                         'voltage','reward','late','controller'])

    print("\nConnecting to hardware...")
    try:
        hw = HardwareInterface(args.port)
        print("✓ Connected")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return

    # ── Model Initialization ──────────────────────────────────────────────────
    print("\nInitializing RL model...")

    class HardwareShapeEnv(gym.Env):
        def __init__(self, include_prev_voltage: bool, frame_stack: int = 1):
            per_frame = 7 if include_prev_voltage else 6
            obs_dim = per_frame * max(1, int(frame_stack))
            self.observation_space = spaces.Box(-1.0, 1.0, (obs_dim,), np.float32)
            self.action_space      = spaces.Box(-1.0, 1.0, (1,), np.float32)
        def reset(self, **kw):
            return np.zeros(self.observation_space.shape, np.float32), {}
        def step(self, a):
            return np.zeros(self.observation_space.shape, np.float32), 0.0, False, False, {}

    _dummy_env = DummyVecEnv([
        lambda: HardwareShapeEnv(args.include_prev_voltage, args.frame_stack)
    ])

    AlgoClass = TQC if args.use_droq else SAC


    policy_kwargs = dict(n_critics=args.n_critics, n_quantiles=15) if args.use_droq else None
    algo_kwargs = dict(top_quantiles_to_drop_per_net=2) if args.use_droq else {}


    per_replay_kwargs = dict(
        replay_buffer_class=PrioritizedReplayBuffer,
        replay_buffer_kwargs=dict(alpha=args.per_alpha),
    ) if args.use_per else {}

    def _migrate_buffer_to_per(m):
        """Copy existing ReplayBuffer storage into a PER buffer in place."""
        if isinstance(m.replay_buffer, PrioritizedReplayBuffer):
            return
        old = m.replay_buffer
        new = PrioritizedReplayBuffer(
            buffer_size=old.buffer_size,
            observation_space=old.observation_space,
            action_space=old.action_space,
            device=old.device,
            n_envs=getattr(old, 'n_envs', 1),
            alpha=args.per_alpha,
        )
        # Copy raw storage
        new.observations[:]      = old.observations
        new.next_observations[:] = old.next_observations
        new.actions[:]           = old.actions
        new.rewards[:]           = old.rewards
        new.dones[:]             = old.dones
        new.timeouts[:]          = old.timeouts
        new.pos                  = old.pos
        new.full                 = old.full
        # Seed priorities for all currently-occupied slots at max priority
        occupied = old.buffer_size if old.full else old.pos
        for i in range(occupied):
            new._set_priority(i, 1.0)
        m.replay_buffer = new
        print(f"✓ Migrated replay buffer to PER ({occupied} transitions)")

    if args.resume:
        stem  = args.resume.replace('.zip', '')
        model = AlgoClass.load(stem, env=_dummy_env, device=device)
        buf_path = stem + '_buffer.pkl'
        if os.path.exists(buf_path):
            model.load_replay_buffer(buf_path)
            print(f"✓ Loaded buffer: {model.replay_buffer.size()} transitions")
        else:
            print(f"⚠ No buffer at {buf_path} — starting empty")
        if args.use_per:
            _migrate_buffer_to_per(model)
        override_hyperparams(model, lr=args.learning_rate,
                             sde_sample_freq=args.sde_sample_freq)
        print(f"✓ Overrode LR={args.learning_rate:g}, sde_sample_freq={args.sde_sample_freq}")

    elif args.model:
        model = AlgoClass.load(args.model.replace('.zip', ''), env=_dummy_env, device=device)
        if args.use_per:
            _migrate_buffer_to_per(model)
        override_hyperparams(model, lr=args.learning_rate,
                             sde_sample_freq=args.sde_sample_freq)
        print(f"✓ Loaded model (no buffer), overrode LR and SDE freq")

    else:
        model = AlgoClass(
            "MlpPolicy",
            _dummy_env,
            learning_rate=float(args.learning_rate),
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            use_sde=True,
            sde_sample_freq=args.sde_sample_freq,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
            **per_replay_kwargs,
            **algo_kwargs  # Unpack the algorithm-specific arguments here
        )

    from stable_baselines3.common.logger import configure
    model.set_logger(configure(folder=None, format_strings=[]))
    model.batch_size = args.batch_size

    if args.freeze_entropy is not None:
        fixed = float(args.freeze_entropy)
        model.ent_coef_optimizer = None
        model.ent_coef_tensor = torch.tensor(fixed, device=model.device)
        if hasattr(model, 'log_ent_coef') and model.log_ent_coef is not None:
            model.log_ent_coef.data = torch.log(torch.tensor(fixed, device=model.device))
            model.log_ent_coef.requires_grad = False
        print(f"✓ Entropy coefficient frozen at {fixed}")

    reset_sde_noise_on_device(model, batch_size=1)
    print("✓ Model ready")

    # ── Dynamics model (Change 2) ─────────────────────────────────────────────
    dynamics = None
    if args.use_model_based:
        dynamics = DynamicsModel(per_frame_dim=args.per_frame_dim,
                                 frame_stack=args.frame_stack,
                                 act_dim=1, hidden=64, device=str(device))
        print(f"✓ Dynamics model ready (per_frame={args.per_frame_dim}, "
              f"stack={args.frame_stack}, obs_dim={dynamics.obs_dim}, "
              f"act=1, hidden=64)")


    bc_weight = float(args.bc_weight)

    print()

    # ── Control loop state ────────────────────────────────────────────────────
    step                  = 0
    late_count            = 0
    ep_reward             = 0.0
    last_voltage          = 0.0
    prev_voltage          = 0.0
    last_obs              = None
    last_action           = None
    transition_count      = 0
    last_controller_was_sac = False

    frame_deque = deque(maxlen=args.frame_stack)

    gc.disable()

    try:
        input("\n  >>> Hold pendulum UPRIGHT and press Enter to start <<<\n")

        for _ in range(args.warmup_steps):
            hw.send_voltage(0.0)
            time.sleep(0.01)

        t_start = time.perf_counter()

        while args.total_steps == 0 or step < args.total_steps:
            sensor_data = hw.get_sensor_data()
            if sensor_data is None:
                continue

            t_compute_start = time.perf_counter()
            pend_pos, arm_pos, pend_vel, arm_vel = (
                float(sensor_data[0]), float(sensor_data[1]),
                float(sensor_data[2]), float(sensor_data[3]),
            )
            pend_pos = ((pend_pos + np.pi) % (2.0 * np.pi)) - np.pi
            raw_obs = state_to_obs(pend_pos, arm_pos, pend_vel, arm_vel,
                                   last_voltage, args.rl_vcap,
                                   args.include_prev_voltage)

            if len(frame_deque) == 0:
                for _ in range(args.frame_stack):
                    frame_deque.append(raw_obs)
            else:
                frame_deque.append(raw_obs)
            obs = (raw_obs if args.frame_stack == 1
                   else stack_obs(frame_deque))

            reward      = compute_reward(pend_pos, arm_pos, pend_vel, arm_vel,
                                         last_voltage, prev_voltage,
                                         rl_angle_rad=args.rl_angle_rad)
            done        = abs(pend_pos) > args.rl_angle_rad
            motor_cutoff = abs(pend_pos) > args.safe_angle

            ep_reward += reward

            # ── Store SAC transition ─────────────────────────────────────
            if (last_obs is not None and last_action is not None
                    and last_controller_was_sac and not TEST_LQR_ONLY):
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
                    model.train(gradient_steps=args.online_gradient_steps,
                                batch_size=args.batch_size)


                    if bc_weight > 1e-6:
                        run_bc_updates(model, bc_weight, n_steps=1,
                                       batch_size=args.batch_size,
                                       rl_vcap=args.rl_vcap, device=device,
                                       per_frame_dim=args.per_frame_dim)
                        bc_weight *= args.bc_decay ** args.online_gradient_steps

            # ── Safety cutoff — one life per Arduino reset ───────────────
            if motor_cutoff:
                hw.send_voltage(0.0)
                print(f"\n{step:>6d} | FALLEN — |theta|={np.degrees(pend_pos):+.1f}deg | "
                      f"ep_reward={ep_reward:+.1f}")

                if not TEST_LQR_ONLY and model.replay_buffer.size() > args.batch_size:
                    buf_sz = model.replay_buffer.size()
                    scaled = min(args.gradient_steps,
                                 max(50, (buf_sz * 10) // args.batch_size))
                    print(f"         [Training {scaled} SAC steps on {buf_sz} real transitions...]")
                    t0 = time.perf_counter()

                    per_train_tqc(model, gradient_steps=scaled,
                                  batch_size=args.batch_size,
                                  beta_start=args.per_beta_start,
                                  beta_end=args.per_beta_end,
                                  beta_anneal_steps=args.per_beta_anneal_steps)
                    print(f"         [SAC/TQC done in {time.perf_counter()-t0:.1f}s]")


                    if dynamics is not None:
                        print(f"         [Fitting dynamics model ({args.model_train_steps} steps)...]")
                        t0       = time.perf_counter()
                        dyn_loss = dynamics.fit(
                            model.replay_buffer,
                            batch_size=min(args.batch_size, 256),
                            n_steps=args.model_train_steps,
                        )
                        print(f"         [Dynamics fit loss={dyn_loss:.5f} "
                              f"in {time.perf_counter()-t0:.1f}s]")

                        print(f"         [Generating synthetic rollouts "
                              f"(seeds={args.model_rollout_seeds}, "
                              f"horizon={args.model_rollout_horizon})...]")
                        t0      = time.perf_counter()
                        n_synth = dynamics.generate_synthetic_transitions(
                            replay_buffer=model.replay_buffer,
                            policy=model.policy,
                            n_seeds=args.model_rollout_seeds,
                            horizon=args.model_rollout_horizon,
                            rl_vcap=args.rl_vcap,
                            safe_angle_rad=args.safe_angle,
                        )
                        print(f"         [{n_synth} synthetic transitions added "
                              f"in {time.perf_counter()-t0:.1f}s]")

                        # Extra pass: train on the now-enriched buffer
                        extra = min(scaled // 2, 100)
                        if extra > 0 and model.replay_buffer.size() > args.batch_size:
                            print(f"         [Extra {extra} steps on enriched buffer...]")
                            per_train_tqc(model, gradient_steps=extra,
                                          batch_size=args.batch_size,
                                          beta_start=args.per_beta_start,
                                          beta_end=args.per_beta_end,
                                          beta_anneal_steps=args.per_beta_anneal_steps)


                    if bc_weight > 1e-6:
                        bc_steps = min(scaled // 4, 50)
                        if bc_steps > 0:
                            bc_loss   = run_bc_updates(
                                model, bc_weight, n_steps=bc_steps,
                                batch_size=args.batch_size,
                                rl_vcap=args.rl_vcap, device=device,
                                per_frame_dim=args.per_frame_dim,
                            )
                            bc_weight *= args.bc_decay ** (scaled + bc_steps)
                            print(f"         [BC loss={bc_loss:.5f}, "
                                  f"bc_weight now {bc_weight:.5f}]")

                model.save("models/hardware_ckpt_latest")
                model.save_replay_buffer("models/hardware_ckpt_latest_buffer.pkl")
                print("         [saved: models/hardware_ckpt_latest(.zip, _buffer.pkl)]")

                print("\n         >>> FALLEN — Hold upright & press RESET <<<")

                while True:
                    try:
                        hw.send_voltage(0.0)
                        sensor_data = hw.get_sensor_data()
                        if sensor_data:
                            p_pos = float(sensor_data[0])
                            p_pos = ((p_pos + np.pi) % (2.0 * np.pi)) - np.pi
                            if abs(p_pos) < args.resume_angle:
                                print("         [Hardware Reset & Upright Detected! Resuming...]")
                                break
                    except Exception:
                        try: hw.ser.close()
                        except: pass
                        time.sleep(1.0)
                        while True:
                            try:
                                hw = HardwareInterface(args.port)
                                break
                            except Exception:
                                time.sleep(0.5)

                for _ in range(args.warmup_steps):
                    hw.send_voltage(0.0)
                    time.sleep(0.01)

                last_voltage          = 0.0
                prev_voltage          = 0.0
                last_obs              = None
                last_action           = None
                last_controller_was_sac = False
                ep_reward             = 0.0
                frame_deque.clear()
                if not TEST_LQR_ONLY:
                    reset_sde_noise_on_device(model, batch_size=1)

                continue

            # ── Controller Selection ──────────────────────────────────────
            active_controller = "Proportional"

            if TEST_LQR_ONLY:
                u       = -(K1*arm_pos + K2*arm_vel +
                             K3*pend_pos + K4*pend_vel + K5*last_voltage)
                voltage = float(np.clip(u, -args.vcap, args.vcap))
                action  = np.array([voltage / args.vcap])
            else:
                abs_theta = abs(pend_pos)
                # Proportional voltage (always computed — cheap)
                prop_u = -(K1*arm_pos + K2*arm_vel +
                           K3*pend_pos + K4*pend_vel + K5*last_voltage)
                prop_v = float(np.clip(prop_u, -args.vcap, args.vcap))

                if abs_theta <= args.blend_inner:
                    # ── Pure SAC zone ──
                    active_controller = "SAC"
                    action, _ = model.predict(obs, deterministic=False)
                    sac_v = (float(action[0]) if isinstance(action, np.ndarray)
                             else float(action)) * args.rl_vcap
                    sac_v = float(np.clip(sac_v, -args.rl_voltage_limit,
                                                  args.rl_voltage_limit))
                    voltage = float(apply_deadzone_model(sac_v))

                elif abs_theta < args.rl_angle_rad:
                    # ── Blend zone: smooth SAC → Proportional transition ──
                    active_controller = "SAC_Blend"
                    alpha = 1.0 - (abs_theta - args.blend_inner) / (
                        args.rl_angle_rad - args.blend_inner)
                    action, _ = model.predict(obs, deterministic=False)
                    sac_v = (float(action[0]) if isinstance(action, np.ndarray)
                             else float(action)) * args.rl_vcap
                    sac_v = float(np.clip(sac_v, -args.rl_voltage_limit,
                                                  args.rl_voltage_limit))
                    voltage = float(apply_deadzone_model(
                        alpha * sac_v + (1.0 - alpha) * prop_v))

                else:
                    # ── Pure proportional (outside RL boundary) ──
                    active_controller = "Proportional"
                    voltage = float(apply_deadzone_model(prop_v))
                    action  = np.array([float(np.clip(
                        voltage / args.rl_vcap, -1.0, 1.0))])

            # Store SAC and SAC_Blend transitions (blend zone sees done boundary)
            last_controller_was_sac = (active_controller in ("SAC", "SAC_Blend"))

            hw.send_voltage(float(voltage))
            prev_voltage = last_voltage
            last_voltage = float(voltage)
            last_obs     = obs

            # Store the ACTUALLY applied (post clip + deadzone) action
            applied_norm = float(np.clip(voltage / args.rl_vcap, -1.0, 1.0))
            last_action  = np.array([applied_norm], dtype=np.float32)

            # Latency check
            t_compute_elapsed = time.perf_counter() - t_compute_start
            if t_compute_elapsed > 0.003:
                late_count += 1

            log_writer.writerow([step, pend_pos, arm_pos, pend_vel, arm_vel,
                                  voltage, reward, late_count, active_controller])

            if step % 10 == 0:
                print(f"Step {step:6d} | theta: {np.degrees(pend_pos):+6.1f}deg | "
                      f"theta_dot: {np.degrees(pend_vel):+6.1f}deg/s | "
                      f"alpha: {np.degrees(arm_pos):+6.1f}deg | "
                      f"alpha_dot: {np.degrees(arm_vel):+6.1f}deg/s | "
                      f"V: {voltage:+6.2f} | late: {late_count} | "
                      f"ctrl: {active_controller}")

            if step > 0 and step % 200 == 0:
                buf_size = (model.replay_buffer.size()
                            if hasattr(model, 'replay_buffer') else 0)
                print(f"         [diag] buf={buf_size} | transitions={transition_count} | "
                      f"ep_reward={ep_reward:+.1f} | bc_weight={bc_weight:.5f}")

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
