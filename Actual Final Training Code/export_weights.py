"""
Export a trained SAC/TQC(DroQ) actor to a self-contained C header for MCU / ESP32 deployment.

For the current DroQ hardware checkpoint, the only things needed from
hardware_ckpt_latest.zip are the deterministic actor layers stored in policy.pth:
  - actor.latent_pi Linear layer weights/biases
  - actor.mu Linear layer weights/biases
  - actor.mu Hardtanh clip limits, when present

Not needed for deployment:
  - critic weights
  - target critic weights
  - optimizer states
  - entropy optimizer
  - replay buffer .pkl

Typical current-project usage from the RL folder:
    ../droq_venv/bin/python export_weights.py \
      --model "../delay_droq/RL/Actual Final Training Code/models/hardware_ckpt_latest" \
      --algo tqc \
      --frame-stack 3 \
      --output wireless/droq_weights.h \
      --rl-voltage-limit 2.0

If your wireless plant ESP already applies motor deadzone compensation, use:
    --output-mode pre-deadzone

If the generated voltage goes directly to the motor/PWM layer, use:
    --output-mode post-deadzone
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

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
    TQC = None
    TQC_AVAILABLE = False


@dataclass
class ExportedActor:
    algo_name: str
    layers: List[Tuple[str, np.ndarray, np.ndarray]]
    obs_dim: int
    per_frame_dim: int
    frame_stack: int
    include_prev_voltage: bool
    mu_clip_min: Optional[float]
    mu_clip_max: Optional[float]


class HardwareShapeEnv(gym.Env):
    def __init__(self, obs_dim: int):
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, **kwargs):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}


def strip_zip(path: str) -> str:
    return path[:-4] if path.endswith(".zip") else path


def linear_from_mu(mu):
    """Return (linear_layer, hardtanh_min, hardtanh_max) from actor.mu."""
    if hasattr(mu, "weight"):
        return mu, None, None

    linear = None
    clip_min = None
    clip_max = None
    if hasattr(mu, "__iter__"):
        for layer in mu:
            if linear is None and hasattr(layer, "weight") and hasattr(layer, "bias"):
                linear = layer
            if layer.__class__.__name__ == "Hardtanh":
                clip_min = float(layer.min_val)
                clip_max = float(layer.max_val)
    if linear is None:
        raise RuntimeError(f"Could not find Linear layer inside actor.mu={mu}")
    return linear, clip_min, clip_max


def load_model(model_path: str, algo: str, obs_dim: int):
    env = DummyVecEnv([lambda: HardwareShapeEnv(obs_dim)])
    errors = []
    candidates = []

    if algo in ("auto", "tqc", "droq"):
        if TQC_AVAILABLE:
            candidates.append(("TQC/DroQ", TQC))
        else:
            errors.append("TQC/DroQ unavailable: sb3_contrib is not installed")
    if algo in ("auto", "sac"):
        candidates.append(("SAC", SAC))

    for name, cls in candidates:
        try:
            model = cls.load(strip_zip(model_path), env=env, device="cpu")
            return name, model
        except Exception as exc:
            errors.append(f"{name} load failed: {exc}")

    msg = "Could not load checkpoint.\n" + "\n".join(f"  - {e}" for e in errors)
    raise RuntimeError(msg)


def extract_actor_weights(
    model_path: str,
    algo: str = "auto",
    include_prev_voltage: bool = True,
    frame_stack: int = 3,
) -> ExportedActor:
    """Extract deterministic actor weights from SAC or TQC/DroQ."""
    per_frame_dim = 7 if include_prev_voltage else 6
    obs_dim = per_frame_dim * int(frame_stack)
    algo_name, model = load_model(model_path, algo, obs_dim)

    actor = getattr(model, "actor", None)
    if actor is None:
        actor = getattr(model.policy, "actor", None)
    if actor is None:
        raise RuntimeError("Could not find actor in loaded model")

    layers = []
    print(f"Loaded {algo_name} actor from {model_path}")
    print(f"  Observation dim: {obs_dim} ({frame_stack} x {per_frame_dim})")

    for layer in actor.latent_pi:
        if hasattr(layer, "weight") and hasattr(layer, "bias"):
            w = layer.weight.detach().cpu().numpy().astype(np.float32)
            b = layer.bias.detach().cpu().numpy().astype(np.float32)
            name = f"hidden_{len(layers)}"
            layers.append((name, w, b))
            print(f"  {name}: weight {w.shape}, bias {b.shape}")
        elif layer.__class__.__name__ != "ReLU":
            raise RuntimeError(
                f"Unsupported actor activation/layer {layer}. "
                "The C exporter currently supports Linear + ReLU hidden networks."
            )

    mu_linear, clip_min, clip_max = linear_from_mu(actor.mu)
    w_mu = mu_linear.weight.detach().cpu().numpy().astype(np.float32)
    b_mu = mu_linear.bias.detach().cpu().numpy().astype(np.float32)
    layers.append(("mu", w_mu, b_mu))
    print(f"  mu: weight {w_mu.shape}, bias {b_mu.shape}")
    if clip_min is not None:
        print(f"  mu Hardtanh clip: [{clip_min}, {clip_max}]")

    return ExportedActor(
        algo_name=algo_name,
        layers=layers,
        obs_dim=obs_dim,
        per_frame_dim=per_frame_dim,
        frame_stack=int(frame_stack),
        include_prev_voltage=include_prev_voltage,
        mu_clip_min=clip_min,
        mu_clip_max=clip_max,
    )


def manual_forward(actor: ExportedActor, obs: np.ndarray) -> float:
    x = obs.astype(np.float32)
    for name, w, b in actor.layers[:-1]:
        x = np.maximum(w @ x + b, 0.0).astype(np.float32)
    _, w_mu, b_mu = actor.layers[-1]
    mu = float((w_mu @ x + b_mu)[0])
    if actor.mu_clip_min is not None:
        mu = float(np.clip(mu, actor.mu_clip_min, actor.mu_clip_max))
    return float(np.tanh(mu))


def verify_export(actor: ExportedActor, model_path: str, algo: str):
    _, model = load_model(model_path, algo, actor.obs_dim)
    rng = np.random.default_rng(7)
    max_err = 0.0
    for _ in range(64):
        obs = rng.uniform(-1.0, 1.0, size=(actor.obs_dim,)).astype(np.float32)
        exported = manual_forward(actor, obs)
        predicted, _ = model.predict(obs, deterministic=True)
        err = abs(exported - float(predicted[0]))
        max_err = max(max_err, err)
    print(f"  Verification max |Python predict - exported forward| = {max_err:.8g}")
    if max_err > 2e-5:
        raise RuntimeError("Export verification failed; generated C would not match deterministic policy")


def array_to_c(name: str, arr: np.ndarray) -> str:
    flat = arr.astype(np.float32).flatten()
    lines = []
    for i in range(0, len(flat), 8):
        chunk = flat[i : i + 8]
        lines.append("    " + ", ".join(f"{v: .8f}f" for v in chunk))
    values = ",\n".join(lines)
    if arr.ndim == 1:
        return f"static const float {name}[{arr.shape[0]}] = {{\n{values}\n}};"
    if arr.ndim == 2:
        return (
            f"// Shape: ({arr.shape[0]}, {arr.shape[1]}) stored row-major\n"
            f"static const float {name}[{arr.shape[0] * arr.shape[1]}] = {{\n{values}\n}};"
        )
    raise ValueError(f"Unsupported array rank for {name}: {arr.ndim}")


def c_bool(value: bool) -> str:
    return "1" if value else "0"


def generate_header(
    actor: ExportedActor,
    output_path: str,
    rl_vcap: float,
    rl_voltage_limit: float,
    rl_angle_deg: float,
    safe_angle: float,
    resume_angle_rad: float,
    deadzone: float,
    threshold: float,
    vcap: float,
    blend_inner_frac: float,
    output_mode: str,
):
    hidden_sizes = [w.shape[0] for _, w, _ in actor.layers[:-1]]
    total_params = sum(w.size + b.size for _, w, b in actor.layers)
    output_pre_deadzone = output_mode == "pre-deadzone"

    lines = []
    lines.append("/*")
    lines.append(" * DroQ/TQC/SAC deterministic actor weights - generated by export_weights.py")
    lines.append(f" * Source algo: {actor.algo_name}")
    lines.append(
        f" * Architecture: {actor.obs_dim} -> "
        f"{' -> '.join(str(s) for s in hidden_sizes)} -> 1"
    )
    lines.append(f" * Frame stack: {actor.frame_stack}")
    lines.append(f" * Total parameters: {total_params}")
    lines.append(" */")
    lines.append("")
    lines.append("#ifndef DROQ_WEIGHTS_H")
    lines.append("#define DROQ_WEIGHTS_H")
    lines.append("")
    lines.append("#include <math.h>")
    lines.append("")
    lines.append(f"#define DROQ_OBS_DIM              {actor.obs_dim}")
    lines.append(f"#define DROQ_PER_FRAME_DIM        {actor.per_frame_dim}")
    lines.append(f"#define DROQ_FRAME_STACK          {actor.frame_stack}")
    lines.append(f"#define DROQ_INCLUDE_PREV_VOLTAGE {c_bool(actor.include_prev_voltage)}")
    lines.append(f"#define DROQ_NUM_HIDDEN           {len(hidden_sizes)}")
    for i, hs in enumerate(hidden_sizes):
        lines.append(f"#define DROQ_HIDDEN_{i}_SIZE       {hs}")
    lines.append(f"#define DROQ_MAX_HIDDEN           {max(hidden_sizes) if hidden_sizes else 1}")
    lines.append("")
    lines.append(f"#define RL_VCAP                   {rl_vcap:.6f}f")
    lines.append(f"#define RL_VOLTAGE_LIMIT          {rl_voltage_limit:.6f}f")
    lines.append(f"#define RL_ANGLE_RAD              {np.radians(rl_angle_deg):.8f}f")
    lines.append(f"#define BLEND_INNER_RAD           {(np.radians(rl_angle_deg) * blend_inner_frac):.8f}f")
    lines.append(f"#define SAFE_ANGLE                {safe_angle:.8f}f")
    lines.append(f"#define RESUME_ANGLE              {resume_angle_rad:.8f}f")
    lines.append(f"#define DEADZONE_VAL              {deadzone:.6f}f")
    lines.append(f"#define DEADZONE_THRESH           {threshold:.6f}f")
    lines.append(f"#define PROP_VCAP                 {vcap:.6f}f")
    lines.append("#define VEL_NORM_PEND             15.0f")
    lines.append("#define VEL_NORM_ARM              25.0f")
    lines.append(f"#define DROQ_OUTPUT_PRE_DEADZONE  {c_bool(output_pre_deadzone)}")
    lines.append("")
    lines.append("#define DROQ_CTRL_RL              0")
    lines.append("#define DROQ_CTRL_BLEND           1")
    lines.append("#define DROQ_CTRL_PROPORTIONAL    2")
    lines.append("#define DROQ_CTRL_FALLEN          3")
    lines.append("")
    lines.append("// Proportional catch gains from training code")
    lines.append(f"#define K1 {-4.4681 * 0.25:.8f}f")
    lines.append(f"#define K2 {-1.6210 * 0.50:.8f}f")
    lines.append(f"#define K3 {-39.5037 * 0.75:.8f}f")
    lines.append(f"#define K4 {-5.0402 * 0.60:.8f}f")
    lines.append(f"#define K5 {-0.4387 * 0.50:.8f}f")
    lines.append("")
    if actor.mu_clip_min is not None:
        lines.append(f"#define MU_CLIP_MIN               {actor.mu_clip_min:.8f}f")
        lines.append(f"#define MU_CLIP_MAX               {actor.mu_clip_max:.8f}f")
        lines.append("#define MU_HAS_HARDTANH           1")
    else:
        lines.append("#define MU_CLIP_MIN               -3.4028235e38f")
        lines.append("#define MU_CLIP_MAX                3.4028235e38f")
        lines.append("#define MU_HAS_HARDTANH           0")
    lines.append("")

    lines.append("// Network weights")
    for name, w, b in actor.layers:
        lines.append(array_to_c(f"w_{name}", w))
        lines.append("")
        lines.append(array_to_c(f"b_{name}", b))
        lines.append("")

    lines.append("static float droq_last_applied_voltage = 0.0f;")
    lines.append("static int droq_stack_initialized = 0;")
    lines.append("static float droq_frame_stack[DROQ_OBS_DIM];")
    lines.append("")
    lines.append("static inline float droq_relu(float x) { return x > 0.0f ? x : 0.0f; }")
    lines.append("")
    lines.append("static inline float droq_clamp(float x, float lo, float hi) {")
    lines.append("    if (x < lo) return lo;")
    lines.append("    if (x > hi) return hi;")
    lines.append("    return x;")
    lines.append("}")
    lines.append("")
    lines.append("static inline float droq_wrap_pi(float x) {")
    lines.append("    while (x > (float)M_PI) x -= 2.0f * (float)M_PI;")
    lines.append("    while (x < -(float)M_PI) x += 2.0f * (float)M_PI;")
    lines.append("    return x;")
    lines.append("}")
    lines.append("")
    lines.append("static float droq_deadzone(float voltage) {")
    lines.append("    if (voltage > DEADZONE_THRESH) voltage += DEADZONE_VAL;")
    lines.append("    else if (voltage < -DEADZONE_THRESH) voltage -= DEADZONE_VAL;")
    lines.append("    else voltage = 0.0f;")
    lines.append("    return droq_clamp(voltage, -PROP_VCAP, PROP_VCAP);")
    lines.append("}")
    lines.append("")
    lines.append("static void droq_reset_state(void) {")
    lines.append("    droq_last_applied_voltage = 0.0f;")
    lines.append("    droq_stack_initialized = 0;")
    lines.append("    for (int i = 0; i < DROQ_OBS_DIM; i++) droq_frame_stack[i] = 0.0f;")
    lines.append("}")
    lines.append("")
    lines.append("static void droq_build_raw_obs(float raw[DROQ_PER_FRAME_DIM],")
    lines.append("                               float pend_pos, float arm_pos,")
    lines.append("                               float pend_vel, float arm_vel) {")
    lines.append("    raw[0] = cosf(pend_pos);")
    lines.append("    raw[1] = sinf(pend_pos);")
    lines.append("    raw[2] = cosf(arm_pos);")
    lines.append("    raw[3] = sinf(arm_pos);")
    lines.append("    raw[4] = pend_vel / VEL_NORM_PEND;")
    lines.append("    raw[5] = arm_vel / VEL_NORM_ARM;")
    if actor.include_prev_voltage:
        lines.append("    raw[6] = droq_clamp(droq_last_applied_voltage / RL_VCAP, -1.0f, 1.0f);")
    lines.append("}")
    lines.append("")
    lines.append("static void droq_update_obs_stack(float obs[DROQ_OBS_DIM], const float raw[DROQ_PER_FRAME_DIM]) {")
    lines.append("    if (!droq_stack_initialized) {")
    lines.append("        for (int f = 0; f < DROQ_FRAME_STACK; f++) {")
    lines.append("            for (int j = 0; j < DROQ_PER_FRAME_DIM; j++) {")
    lines.append("                droq_frame_stack[f * DROQ_PER_FRAME_DIM + j] = raw[j];")
    lines.append("            }")
    lines.append("        }")
    lines.append("        droq_stack_initialized = 1;")
    lines.append("    } else {")
    lines.append("        for (int i = 0; i < DROQ_OBS_DIM - DROQ_PER_FRAME_DIM; i++) {")
    lines.append("            droq_frame_stack[i] = droq_frame_stack[i + DROQ_PER_FRAME_DIM];")
    lines.append("        }")
    lines.append("        for (int j = 0; j < DROQ_PER_FRAME_DIM; j++) {")
    lines.append("            droq_frame_stack[DROQ_OBS_DIM - DROQ_PER_FRAME_DIM + j] = raw[j];")
    lines.append("        }")
    lines.append("    }")
    lines.append("    for (int i = 0; i < DROQ_OBS_DIM; i++) obs[i] = droq_frame_stack[i];")
    lines.append("}")
    lines.append("")
    lines.append("static float droq_actor_forward(const float obs[DROQ_OBS_DIM]) {")

    # Hidden layers, explicit for speed and Arduino friendliness.
    previous_name = "obs"
    previous_size = actor.obs_dim
    for idx, (name, w, b) in enumerate(actor.layers[:-1]):
        h_name = f"h{idx}"
        h_size = w.shape[0]
        w_name = f"w_{name}"
        b_name = f"b_{name}"
        lines.append(f"    float {h_name}[{h_size}];")
        lines.append(f"    for (int i = 0; i < {h_size}; i++) {{")
        lines.append(f"        float sum = {b_name}[i];")
        lines.append(f"        for (int j = 0; j < {previous_size}; j++) {{")
        lines.append(f"            sum += {w_name}[i * {previous_size} + j] * {previous_name}[j];")
        lines.append("        }")
        lines.append(f"        {h_name}[i] = droq_relu(sum);")
        lines.append("    }")
        lines.append("")
        previous_name = h_name
        previous_size = h_size

    lines.append("    float mu = b_mu[0];")
    lines.append(f"    for (int j = 0; j < {previous_size}; j++) {{")
    lines.append(f"        mu += w_mu[j] * {previous_name}[j];")
    lines.append("    }")
    lines.append("    if (MU_HAS_HARDTANH) mu = droq_clamp(mu, MU_CLIP_MIN, MU_CLIP_MAX);")
    lines.append("    return tanhf(mu);")
    lines.append("}")
    lines.append("")
    lines.append("static float droq_proportional_pre_voltage(float arm_pos, float arm_vel,")
    lines.append("                                          float pend_pos, float pend_vel) {")
    lines.append("    float u = -(K1 * arm_pos + K2 * arm_vel + K3 * pend_pos + K4 * pend_vel + K5 * droq_last_applied_voltage);")
    lines.append("    return droq_clamp(u, -PROP_VCAP, PROP_VCAP);")
    lines.append("}")
    lines.append("")
    lines.append("static float droq_output_voltage(float pre_deadzone_voltage, float *applied_voltage) {")
    lines.append("    float applied = droq_deadzone(pre_deadzone_voltage);")
    lines.append("    *applied_voltage = applied;")
    lines.append("    if (DROQ_OUTPUT_PRE_DEADZONE) return pre_deadzone_voltage;")
    lines.append("    return applied;")
    lines.append("}")
    lines.append("")
    lines.append("/*")
    lines.append(" * Full control pipeline. Call once per control tick on the controller ESP.")
    lines.append(" * Returns voltage command. If DROQ_OUTPUT_PRE_DEADZONE=1, the plant/PWM layer")
    lines.append(" * should apply deadzone compensation. Otherwise this returns the final voltage.")
    lines.append(" */")
    lines.append("static float droq_control_step(float pend_pos, float arm_pos,")
    lines.append("                               float pend_vel, float arm_vel,")
    lines.append("                               int *controller) {")
    lines.append("    pend_pos = droq_wrap_pi(pend_pos);")
    lines.append("    float abs_pend = fabsf(pend_pos);")
    lines.append("    if (abs_pend > SAFE_ANGLE) {")
    lines.append("        if (controller) *controller = DROQ_CTRL_FALLEN;")
    lines.append("        droq_reset_state();")
    lines.append("        return 0.0f;")
    lines.append("    }")
    lines.append("")
    lines.append("    float raw[DROQ_PER_FRAME_DIM];")
    lines.append("    float obs[DROQ_OBS_DIM];")
    lines.append("    droq_build_raw_obs(raw, pend_pos, arm_pos, pend_vel, arm_vel);")
    lines.append("    droq_update_obs_stack(obs, raw);")
    lines.append("")
    lines.append("    float prop_pre = droq_proportional_pre_voltage(arm_pos, arm_vel, pend_pos, pend_vel);")
    lines.append("    float selected_pre = prop_pre;")
    lines.append("")
    lines.append("    if (abs_pend <= BLEND_INNER_RAD) {")
    lines.append("        if (controller) *controller = DROQ_CTRL_RL;")
    lines.append("        float action = droq_actor_forward(obs);")
    lines.append("        selected_pre = droq_clamp(action * RL_VCAP, -RL_VOLTAGE_LIMIT, RL_VOLTAGE_LIMIT);")
    lines.append("    } else if (abs_pend < RL_ANGLE_RAD) {")
    lines.append("        if (controller) *controller = DROQ_CTRL_BLEND;")
    lines.append("        float alpha = 1.0f - (abs_pend - BLEND_INNER_RAD) / (RL_ANGLE_RAD - BLEND_INNER_RAD);")
    lines.append("        float action = droq_actor_forward(obs);")
    lines.append("        float rl_pre = droq_clamp(action * RL_VCAP, -RL_VOLTAGE_LIMIT, RL_VOLTAGE_LIMIT);")
    lines.append("        selected_pre = alpha * rl_pre + (1.0f - alpha) * prop_pre;")
    lines.append("    } else {")
    lines.append("        if (controller) *controller = DROQ_CTRL_PROPORTIONAL;")
    lines.append("        selected_pre = prop_pre;")
    lines.append("    }")
    lines.append("")
    lines.append("    float applied = 0.0f;")
    lines.append("    float command = droq_output_voltage(selected_pre, &applied);")
    lines.append("    droq_last_applied_voltage = applied;")
    lines.append("    return command;")
    lines.append("}")
    lines.append("")
    lines.append("#endif // DROQ_WEIGHTS_H")
    lines.append("")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"\nHeader written to: {output}")

    print(f"  Architecture: {actor.obs_dim} -> {' -> '.join(str(s) for s in hidden_sizes)} -> 1")
    print(f"  Parameters: {total_params:,} floats = {total_params * 4 / 1024:.1f} KB")
    print(f"  Output mode: {output_mode}")


def main():
    parser = argparse.ArgumentParser(description="Export SAC/TQC(DroQ) actor to C header")
    parser.add_argument("--model", required=True, help="Path to hardware_ckpt_latest or .zip")
    parser.add_argument("--output", default="droq_weights.h", help="Output C header path")
    parser.add_argument("--algo", choices=["auto", "tqc", "droq", "sac"], default="auto")
    parser.add_argument("--frame-stack", type=int, default=3, help="Must match training; current DroQ default is 3")
    parser.add_argument("--include-prev-voltage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl-vcap", type=float, default=12.0)
    parser.add_argument("--rl-voltage-limit", type=float, default=2.0)
    parser.add_argument("--rl-angle", type=float, default=4.0, help="RL boundary in degrees")
    parser.add_argument("--blend-inner-frac", type=float, default=0.7)
    parser.add_argument("--safe-angle", type=float, default=0.40)
    parser.add_argument("--resume-angle", type=float, default=float(np.radians(4)))
    parser.add_argument("--deadzone", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--vcap", type=float, default=12.0)
    parser.add_argument(
        "--output-mode",
        choices=["post-deadzone", "pre-deadzone"],
        default="post-deadzone",
        help=(
            "post-deadzone matches train/deploy_RL direct motor voltage. "
            "pre-deadzone is for a wireless plant sketch that applies deadzone itself."
        ),
    )
    parser.add_argument("--no-verify", action="store_true", help="Skip Python-vs-export forward check")
    args = parser.parse_args()

    if args.frame_stack < 1:
        raise ValueError("--frame-stack must be >= 1")

    print("Extracting deterministic actor weights...")
    actor = extract_actor_weights(
        args.model,
        algo=args.algo,
        include_prev_voltage=args.include_prev_voltage,
        frame_stack=args.frame_stack,
    )

    if not args.no_verify:
        print("\nVerifying exported forward pass...")
        verify_export(actor, args.model, args.algo)

    print("\nGenerating C header...")
    generate_header(
        actor,
        args.output,
        rl_vcap=args.rl_vcap,
        rl_voltage_limit=args.rl_voltage_limit,
        rl_angle_deg=args.rl_angle,
        safe_angle=args.safe_angle,
        resume_angle_rad=args.resume_angle,
        deadzone=args.deadzone,
        threshold=args.threshold,
        vcap=args.vcap,
        blend_inner_frac=args.blend_inner_frac,
        output_mode=args.output_mode,
    )

    print("\nWhat was extracted from the checkpoint:")
    print("  actor.latent_pi Linear weights/biases")
    print("  actor.mu Linear weights/biases")
    print("  actor.mu Hardtanh clip, if present")
    print("\nWhat was intentionally ignored:")
    print("  critics, target critics, optimizers, entropy optimizer, replay buffer")


if __name__ == "__main__":
    main()
