#!/usr/bin/env python3

"""Conservative ROS 2 policy runner for the real so101 robot."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from tf2_ros import Buffer, TransformException, TransformListener
except ImportError as exc:  # pragma: no cover - depends on ROS 2 runtime
    raise SystemExit(
        "ROS 2 Python packages are required. Source your ROS 2 environment before running this script. "
        f"Original import error: {exc}"
    ) from exc


CONTROLLED_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
ACTION_SCALE = np.array([0.08, 0.08, 0.08, 0.10, 0.13, 0.06], dtype=np.float32)
DEFAULT_ACTION_SCALE_GAIN = 0.042
DEFAULT_TARGET_POS = np.array([0.180, 0.000, 0.200], dtype=np.float32)
TICKS_PER_REVOLUTION = 4096.0
TICKS_PER_RADIAN = TICKS_PER_REVOLUTION / (2.0 * np.pi)
RADIANS_PER_TICK = 1.0 / TICKS_PER_RADIAN
DEFAULT_JOINT_LOWER_LIMITS = np.array(
    [-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.174533],
    dtype=np.float32,
)
DEFAULT_JOINT_UPPER_LIMITS = np.array(
    [1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533],
    dtype=np.float32,
)
DEFAULT_MIN_EFFECTIVE_TICKS = np.array([6.0, 8.0, 8.0, 4.0, 3.0, 2.0], dtype=np.float32)
DEFAULT_MOTION_EPSILON_TICKS = np.array([1.5, 1.5, 1.5, 1.0, 1.0, 1.0], dtype=np.float32)
DEFAULT_STUCK_CURRENT_THRESHOLDS = np.array([8.0, 35.0, 20.0, 10.0, 8.0, 5.0], dtype=np.float32)
DEFAULT_STUCK_BOOST_TICKS = np.array([12.0, 16.0, 16.0, 6.0, 5.0, 3.0], dtype=np.float32)
DEFAULT_DEADBAND_ENABLE_MASK = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@dataclass
class WrapperTelemetry:
    """Debug information produced by the hardware-aware action wrapper."""

    policy_joint_delta: np.ndarray
    applied_joint_delta: np.ndarray
    observed_joint_delta: np.ndarray
    current_raw: np.ndarray
    effective_min_ticks: np.ndarray
    deadband_mask: np.ndarray
    stuck_mask: np.ndarray
    boost_mask: np.ndarray


class PolicyBackend(Protocol):
    """Inference backend interface."""

    def infer(self, observation: np.ndarray) -> np.ndarray:
        """Run policy inference for a single observation."""


class OnnxPolicyBackend:
    """ONNX Runtime backend."""

    def __init__(self, model_path: Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("onnxruntime is required to load an ONNX policy.") from exc

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def infer(self, observation: np.ndarray) -> np.ndarray:
        outputs = self._session.run([self._output_name], {self._input_name: observation[None, :]})
        return np.asarray(outputs[0][0], dtype=np.float32)


class TorchPolicyBackend:
    """TorchScript backend."""

    def __init__(self, model_path: Path):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("torch is required to load a TorchScript policy.") from exc

        self._torch = torch
        self._model = torch.jit.load(str(model_path), map_location="cpu")
        self._model.eval()

    def infer(self, observation: np.ndarray) -> np.ndarray:
        with self._torch.inference_mode():
            tensor = self._torch.from_numpy(observation[None, :])
            outputs = self._model(tensor)
        return outputs[0].detach().cpu().numpy().astype(np.float32)


@dataclass
class RunnerConfig:
    """Runtime configuration for the real robot policy node."""

    policy_path: Path
    joint_names: list[str] = field(default_factory=lambda: list(CONTROLLED_JOINTS))
    target_pos: np.ndarray = field(default_factory=lambda: DEFAULT_TARGET_POS.copy())
    joint_lower_limits: np.ndarray = field(default_factory=lambda: DEFAULT_JOINT_LOWER_LIMITS.copy())
    joint_upper_limits: np.ndarray = field(default_factory=lambda: DEFAULT_JOINT_UPPER_LIMITS.copy())
    joint_state_topic: str = "/joint_states"
    current_topic: str = "/follower/joint_currents"
    current_state_field: str = "effort"
    command_topic: str = "/so101/policy_joint_command"
    tf_parent_frame: str = "base_link"
    tf_child_frame: str = "gripper_frame_link"
    control_period: float = 1.0 / 60.0
    action_scale_gain: float = DEFAULT_ACTION_SCALE_GAIN
    max_delta_scale: float = 0.5
    stale_timeout_s: float = 0.2
    current_stale_timeout_s: float = 0.2
    status_log_period_s: float = 0.5
    initial_joint_target: np.ndarray | None = None
    enable_action_wrapper: bool = False
    enable_deadband_compensation: bool = True
    enable_stuck_boost: bool = True
    wrapper_far_distance_m: float = 0.05
    wrapper_near_distance_m: float = 0.01
    near_goal_tick_scale: float = 0.35
    deadband_near_goal_tick_scale: float = 0.10
    min_effective_ticks: np.ndarray = field(default_factory=lambda: DEFAULT_MIN_EFFECTIVE_TICKS.copy())
    deadband_enable_mask: np.ndarray = field(default_factory=lambda: DEFAULT_DEADBAND_ENABLE_MASK.copy())
    deadband_persistence_steps: int = 3
    motion_epsilon_ticks: np.ndarray = field(default_factory=lambda: DEFAULT_MOTION_EPSILON_TICKS.copy())
    stuck_current_thresholds: np.ndarray = field(default_factory=lambda: DEFAULT_STUCK_CURRENT_THRESHOLDS.copy())
    stuck_boost_ticks: np.ndarray = field(default_factory=lambda: DEFAULT_STUCK_BOOST_TICKS.copy())
    stuck_consecutive_steps: int = 3
    record_rollout: bool = False
    rollout_dir: Path = Path("logs/real_rollouts")
    rollout_name: str | None = None


@dataclass
class RobotState:
    """Latest robot state required to build the observation."""

    joint_pos: np.ndarray | None = None
    joint_vel: np.ndarray | None = None
    joint_current: np.ndarray | None = None


class RealRolloutRecorder:
    """Collects real-robot rollout data and stores it as a compressed NumPy archive."""

    def __init__(self, cfg: RunnerConfig):
        self._cfg = cfg
        self._records: dict[str, list[np.ndarray | float | int]] = {
            "step": [],
            "timestamp_ns": [],
            "observation": [],
            "prev_action_input": [],
            "action": [],
            "policy_joint_delta": [],
            "joint_delta": [],
            "joint_observed_delta": [],
            "joint_target_before": [],
            "joint_target_after": [],
            "joint_pos": [],
            "joint_vel": [],
            "joint_current_raw": [],
            "wrapper_effective_min_ticks": [],
            "wrapper_deadband_mask": [],
            "wrapper_stuck_mask": [],
            "wrapper_boost_mask": [],
            "ee_pos": [],
            "target_pos": [],
            "distance": [],
        }

    def append(
        self,
        *,
        timestamp_ns: int,
        observation: np.ndarray,
        prev_action_input: np.ndarray,
        action: np.ndarray,
        policy_joint_delta: np.ndarray,
        joint_delta: np.ndarray,
        observed_joint_delta: np.ndarray,
        joint_target_before: np.ndarray,
        joint_target_after: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        joint_current_raw: np.ndarray,
        wrapper_effective_min_ticks: np.ndarray,
        wrapper_deadband_mask: np.ndarray,
        wrapper_stuck_mask: np.ndarray,
        wrapper_boost_mask: np.ndarray,
        ee_pos: np.ndarray,
        target_pos: np.ndarray,
        distance: float,
    ) -> None:
        step_index = len(self._records["step"])
        self._records["step"].append(step_index)
        self._records["timestamp_ns"].append(timestamp_ns)
        self._records["observation"].append(observation.copy())
        self._records["prev_action_input"].append(prev_action_input.copy())
        self._records["action"].append(action.copy())
        self._records["policy_joint_delta"].append(policy_joint_delta.copy())
        self._records["joint_delta"].append(joint_delta.copy())
        self._records["joint_observed_delta"].append(observed_joint_delta.copy())
        self._records["joint_target_before"].append(joint_target_before.copy())
        self._records["joint_target_after"].append(joint_target_after.copy())
        self._records["joint_pos"].append(joint_pos.copy())
        self._records["joint_vel"].append(joint_vel.copy())
        self._records["joint_current_raw"].append(joint_current_raw.copy())
        self._records["wrapper_effective_min_ticks"].append(wrapper_effective_min_ticks.copy())
        self._records["wrapper_deadband_mask"].append(wrapper_deadband_mask.copy())
        self._records["wrapper_stuck_mask"].append(wrapper_stuck_mask.copy())
        self._records["wrapper_boost_mask"].append(wrapper_boost_mask.copy())
        self._records["ee_pos"].append(ee_pos.copy())
        self._records["target_pos"].append(target_pos.copy())
        self._records["distance"].append(distance)

    def save(self) -> Path | None:
        if not self._records["step"]:
            return None

        self._cfg.rollout_dir.mkdir(parents=True, exist_ok=True)
        rollout_stem = self._cfg.rollout_name or datetime.now().strftime("real_rollout_%Y-%m-%d_%H-%M-%S")
        rollout_path = self._cfg.rollout_dir / f"{rollout_stem}.npz"

        metadata = {
            "source": "real_policy_runner",
            "joint_names": list(self._cfg.joint_names),
            "joint_state_topic": self._cfg.joint_state_topic,
            "current_topic": self._cfg.current_topic,
            "command_topic": self._cfg.command_topic,
            "tf_parent_frame": self._cfg.tf_parent_frame,
            "tf_child_frame": self._cfg.tf_child_frame,
            "control_period": self._cfg.control_period,
            "action_scale_gain": self._cfg.action_scale_gain,
            "max_delta_scale": self._cfg.max_delta_scale,
            "enable_action_wrapper": self._cfg.enable_action_wrapper,
            "enable_deadband_compensation": self._cfg.enable_deadband_compensation,
            "enable_stuck_boost": self._cfg.enable_stuck_boost,
            "wrapper_far_distance_m": self._cfg.wrapper_far_distance_m,
            "wrapper_near_distance_m": self._cfg.wrapper_near_distance_m,
            "near_goal_tick_scale": self._cfg.near_goal_tick_scale,
            "deadband_near_goal_tick_scale": self._cfg.deadband_near_goal_tick_scale,
            "min_effective_ticks": self._cfg.min_effective_ticks.tolist(),
            "deadband_enable_mask": self._cfg.deadband_enable_mask.tolist(),
            "deadband_persistence_steps": self._cfg.deadband_persistence_steps,
            "motion_epsilon_ticks": self._cfg.motion_epsilon_ticks.tolist(),
            "stuck_current_thresholds": self._cfg.stuck_current_thresholds.tolist(),
            "stuck_boost_ticks": self._cfg.stuck_boost_ticks.tolist(),
            "stuck_consecutive_steps": self._cfg.stuck_consecutive_steps,
        }

        arrays: dict[str, np.ndarray] = {"metadata_json": np.asarray(json.dumps(metadata))}
        for key, values in self._records.items():
            first_value = values[0]
            if isinstance(first_value, np.ndarray):
                arrays[key] = np.stack(values).astype(np.float32)
            else:
                arrays[key] = np.asarray(values)

        np.savez_compressed(rollout_path, **arrays)
        return rollout_path


class So101RealPolicyRunner(Node):
    """ROS 2 node that runs the trained so101 policy against a real robot interface."""

    def __init__(self, config: RunnerConfig):
        super().__init__("so101_real_policy_runner")
        self._cfg = config
        self._policy = self._load_policy(config.policy_path)
        self._state = RobotState()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._prev_action = np.zeros(len(config.joint_names), dtype=np.float32)
        self._joint_pos_target = (
            config.initial_joint_target.copy()
            if config.initial_joint_target is not None
            else np.zeros(len(config.joint_names), dtype=np.float32)
        )
        self._last_joint_state_receive_time = None
        self._last_current_receive_time = None
        self._missing_tf_warned = False
        self._last_distance_to_goal = None
        self._last_ee_pos = None
        self._last_status_label = "waiting"
        self._prev_joint_pos_for_wrapper = None
        self._deadband_counts = np.zeros(len(config.joint_names), dtype=np.int32)
        self._deadband_last_sign = np.zeros(len(config.joint_names), dtype=np.int32)
        self._stuck_counts = np.zeros(len(config.joint_names), dtype=np.int32)
        self._last_wrapper_telemetry = WrapperTelemetry(
            policy_joint_delta=np.zeros(len(config.joint_names), dtype=np.float32),
            applied_joint_delta=np.zeros(len(config.joint_names), dtype=np.float32),
            observed_joint_delta=np.zeros(len(config.joint_names), dtype=np.float32),
            current_raw=np.zeros(len(config.joint_names), dtype=np.float32),
            effective_min_ticks=np.zeros(len(config.joint_names), dtype=np.float32),
            deadband_mask=np.zeros(len(config.joint_names), dtype=np.float32),
            stuck_mask=np.zeros(len(config.joint_names), dtype=np.float32),
            boost_mask=np.zeros(len(config.joint_names), dtype=np.float32),
        )
        self._rollout_recorder = RealRolloutRecorder(config) if config.record_rollout else None

        self.create_subscription(JointState, config.joint_state_topic, self._on_joint_state, 10)
        self.create_subscription(JointState, config.current_topic, self._on_joint_current, 10)
        self._command_pub = self.create_publisher(JointState, config.command_topic, 10)
        self.create_timer(config.control_period, self._control_step)
        self.create_timer(config.status_log_period_s, self._status_log_step)

        self.get_logger().info(f"Loaded policy from: {config.policy_path}")
        self.get_logger().info(
            "Waiting for joint states on '%s' and TF '%s -> %s'."
            % (config.joint_state_topic, config.tf_parent_frame, config.tf_child_frame)
        )
        if config.enable_action_wrapper:
            self.get_logger().info(
                "Hardware-aware wrapper enabled using current topic '%s' (%s field)."
                % (config.current_topic, config.current_state_field)
            )

    def _load_policy(self, model_path: Path) -> PolicyBackend:
        suffix = model_path.suffix.lower()
        if suffix == ".onnx":
            return OnnxPolicyBackend(model_path)
        if suffix in {".pt", ".pth"}:
            return TorchPolicyBackend(model_path)
        raise ValueError(f"Unsupported policy format: {model_path}")

    def _on_joint_state(self, msg: JointState) -> None:
        joint_pos, joint_vel = self._extract_joint_state(msg)
        if joint_pos is None or joint_vel is None:
            return
        if not self._is_finite_array(joint_pos, "joint positions"):
            return
        if not self._is_finite_array(joint_vel, "joint velocities"):
            return
        self._state.joint_pos = joint_pos
        self._state.joint_vel = joint_vel
        self._last_joint_state_receive_time = self.get_clock().now()
        if np.allclose(self._joint_pos_target, 0.0) and self._cfg.initial_joint_target is None:
            self._joint_pos_target = joint_pos.copy()
        if self._prev_joint_pos_for_wrapper is None:
            self._prev_joint_pos_for_wrapper = joint_pos.copy()

    def _on_joint_current(self, msg: JointState) -> None:
        joint_current = self._extract_joint_signal(msg, self._cfg.current_state_field)
        if joint_current is None:
            return
        if not self._is_finite_array(joint_current, "joint current"):
            return
        self._state.joint_current = joint_current
        self._last_current_receive_time = self.get_clock().now()

    def _extract_joint_state(self, msg: JointState) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not msg.name or len(msg.position) != len(msg.name):
            self.get_logger().warning("Received malformed JointState. Ignoring message.")
            return None, None
        if not msg.velocity:
            joint_vel = np.zeros(len(self._cfg.joint_names), dtype=np.float32)
        else:
            joint_vel = None

        positions = []
        velocities = []
        name_to_index = {name: index for index, name in enumerate(msg.name)}
        missing_names = [name for name in self._cfg.joint_names if name not in name_to_index]
        if missing_names:
            self.get_logger().warning(f"JointState is missing required joints: {missing_names}")
            return None, None

        for local_index, name in enumerate(self._cfg.joint_names):
            index = name_to_index[name]
            positions.append(msg.position[index])
            if joint_vel is None:
                velocities.append(msg.velocity[index])
            else:
                velocities.append(joint_vel[local_index])
        return np.asarray(positions, dtype=np.float32), np.asarray(velocities, dtype=np.float32)

    def _extract_joint_signal(self, msg: JointState, field_name: str) -> np.ndarray | None:
        if not msg.name:
            self.get_logger().warning("Received JointState without names for signal extraction.")
            return None

        field_lookup = {
            "position": msg.position,
            "velocity": msg.velocity,
            "effort": msg.effort,
        }
        values = field_lookup.get(field_name)
        if values is None or len(values) != len(msg.name):
            for fallback_field in ("effort", "position", "velocity"):
                fallback_values = field_lookup[fallback_field]
                if fallback_values and len(fallback_values) == len(msg.name):
                    values = fallback_values
                    break
        if values is None or len(values) != len(msg.name):
            self.get_logger().warning("Received JointState without usable '%s' values." % field_name)
            return None

        name_to_index = {name: index for index, name in enumerate(msg.name)}
        missing_names = [name for name in self._cfg.joint_names if name not in name_to_index]
        if missing_names:
            self.get_logger().warning(f"JointState signal is missing required joints: {missing_names}")
            return None
        return np.asarray([values[name_to_index[name]] for name in self._cfg.joint_names], dtype=np.float32)

    def _control_step(self) -> None:
        if self._state.joint_pos is None or self._state.joint_vel is None:
            return
        if not self._joint_state_is_fresh():
            self.get_logger().warning("Joint state timed out. Skipping policy step.")
            return

        observation = self._build_observation()
        if observation is None:
            return
        if not self._is_finite_array(observation, "observation"):
            return

        target_delta = observation[6:9]
        distance = float(np.linalg.norm(target_delta))
        ee_pos = self._cfg.target_pos - target_delta
        self._last_distance_to_goal = distance
        self._last_ee_pos = ee_pos.copy()
        prev_action_input = self._prev_action.copy()
        action = self._policy.infer(observation)
        if not self._is_finite_array(action, "policy action"):
            return
        action = np.clip(action.astype(np.float32), -1.0, 1.0)
        policy_joint_delta = action * ACTION_SCALE * self._cfg.action_scale_gain
        policy_joint_delta = np.clip(
            policy_joint_delta,
            -ACTION_SCALE * self._cfg.max_delta_scale,
            ACTION_SCALE * self._cfg.max_delta_scale,
        )
        joint_target_before = self._joint_pos_target.copy()
        joint_delta, wrapper_telemetry = self._apply_action_wrapper(policy_joint_delta, distance)

        desired_target = np.clip(
            self._joint_pos_target + joint_delta,
            self._cfg.joint_lower_limits,
            self._cfg.joint_upper_limits,
        )
        self._joint_pos_target = desired_target.astype(np.float32)
        if not self._is_finite_array(self._joint_pos_target, "joint position target"):
            return
        self._publish_joint_target(self._joint_pos_target)
        if self._rollout_recorder is not None:
            now = self.get_clock().now().nanoseconds
            self._rollout_recorder.append(
                timestamp_ns=now,
                observation=observation,
                prev_action_input=prev_action_input,
                action=action,
                policy_joint_delta=wrapper_telemetry.policy_joint_delta,
                joint_delta=joint_delta,
                observed_joint_delta=wrapper_telemetry.observed_joint_delta,
                joint_target_before=joint_target_before,
                joint_target_after=self._joint_pos_target,
                joint_pos=self._state.joint_pos,
                joint_vel=self._state.joint_vel,
                joint_current_raw=wrapper_telemetry.current_raw,
                wrapper_effective_min_ticks=wrapper_telemetry.effective_min_ticks,
                wrapper_deadband_mask=wrapper_telemetry.deadband_mask,
                wrapper_stuck_mask=wrapper_telemetry.stuck_mask,
                wrapper_boost_mask=wrapper_telemetry.boost_mask,
                ee_pos=ee_pos,
                target_pos=self._cfg.target_pos,
                distance=distance,
            )
        self._prev_action = action
        self._prev_joint_pos_for_wrapper = self._state.joint_pos.copy()

    def _joint_state_is_fresh(self) -> bool:
        if self._last_joint_state_receive_time is None:
            return False
        elapsed = (self.get_clock().now() - self._last_joint_state_receive_time).nanoseconds * 1e-9
        return elapsed <= self._cfg.stale_timeout_s

    def _current_state_is_fresh(self) -> bool:
        if self._last_current_receive_time is None:
            return False
        elapsed = (self.get_clock().now() - self._last_current_receive_time).nanoseconds * 1e-9
        return elapsed <= self._cfg.current_stale_timeout_s

    def _apply_action_wrapper(self, policy_joint_delta: np.ndarray, distance_to_goal: float) -> tuple[np.ndarray, WrapperTelemetry]:
        observed_joint_delta = np.zeros(len(self._cfg.joint_names), dtype=np.float32)
        if self._state.joint_pos is not None and self._prev_joint_pos_for_wrapper is not None:
            observed_joint_delta = self._state.joint_pos - self._prev_joint_pos_for_wrapper

        current_raw = np.zeros(len(self._cfg.joint_names), dtype=np.float32)
        current_is_fresh = self._current_state_is_fresh()
        if current_is_fresh and self._state.joint_current is not None:
            current_raw = self._state.joint_current.copy()

        effective_min_ticks = np.zeros(len(self._cfg.joint_names), dtype=np.float32)
        deadband_mask = np.zeros(len(self._cfg.joint_names), dtype=np.float32)
        stuck_mask = np.zeros(len(self._cfg.joint_names), dtype=np.float32)
        boost_mask = np.zeros(len(self._cfg.joint_names), dtype=np.float32)

        if not self._cfg.enable_action_wrapper:
            telemetry = WrapperTelemetry(
                policy_joint_delta=policy_joint_delta.copy(),
                applied_joint_delta=policy_joint_delta.copy(),
                observed_joint_delta=observed_joint_delta.astype(np.float32),
                current_raw=current_raw,
                effective_min_ticks=effective_min_ticks,
                deadband_mask=deadband_mask,
                stuck_mask=stuck_mask,
                boost_mask=boost_mask,
            )
            self._last_wrapper_telemetry = telemetry
            self._deadband_counts.fill(0)
            self._deadband_last_sign.fill(0)
            self._stuck_counts.fill(0)
            self._last_status_label = "tracking"
            return policy_joint_delta.copy(), telemetry

        deadband_scale = self._compute_distance_scale(distance_to_goal, self._cfg.deadband_near_goal_tick_scale)
        boost_scale = self._compute_distance_scale(distance_to_goal, self._cfg.near_goal_tick_scale)
        effective_min_ticks = self._cfg.min_effective_ticks * deadband_scale
        effective_boost_ticks = self._cfg.stuck_boost_ticks * boost_scale

        commanded_ticks = policy_joint_delta * TICKS_PER_RADIAN
        applied_ticks = commanded_ticks.copy()
        commanded_abs_ticks = np.abs(commanded_ticks)
        commanded_nonzero = commanded_abs_ticks > 1.0e-3
        commanded_sign = np.sign(commanded_ticks).astype(np.int32)
        observed_abs_ticks = np.abs(observed_joint_delta) * TICKS_PER_RADIAN
        motion_small = observed_abs_ticks <= self._cfg.motion_epsilon_ticks
        deadband_enabled = self._cfg.deadband_enable_mask > 0.5

        if self._cfg.enable_deadband_compensation:
            deadband_candidate = deadband_enabled & commanded_nonzero & (commanded_abs_ticks < effective_min_ticks) & motion_small
            same_direction = deadband_candidate & (commanded_sign == self._deadband_last_sign)
            self._deadband_counts = np.where(deadband_candidate, np.where(same_direction, self._deadband_counts + 1, 1), 0)
            self._deadband_last_sign = np.where(deadband_candidate, commanded_sign, 0)
            deadband_condition = deadband_candidate & (self._deadband_counts >= self._cfg.deadband_persistence_steps)
            applied_ticks[deadband_condition] = np.sign(commanded_ticks[deadband_condition]) * effective_min_ticks[deadband_condition]
            deadband_mask[deadband_condition] = 1.0
        else:
            self._deadband_counts.fill(0)
            self._deadband_last_sign.fill(0)

        tracking_error_ticks = np.abs(self._joint_pos_target - self._state.joint_pos) * TICKS_PER_RADIAN
        current_abs = np.abs(current_raw)
        current_high = current_is_fresh & (current_abs >= self._cfg.stuck_current_thresholds)
        command_active = np.abs(applied_ticks) >= np.maximum(1.0, 0.5 * effective_min_ticks)
        tracking_large = tracking_error_ticks >= np.maximum(1.0, effective_min_ticks)

        stuck_candidate = command_active & motion_small & tracking_large & current_high
        self._stuck_counts = np.where(stuck_candidate, self._stuck_counts + 1, 0)
        stuck_mask[stuck_candidate] = 1.0

        if self._cfg.enable_stuck_boost:
            boost_condition = self._stuck_counts >= self._cfg.stuck_consecutive_steps
            applied_ticks[boost_condition] = np.sign(applied_ticks[boost_condition]) * np.maximum(
                np.abs(applied_ticks[boost_condition]),
                effective_boost_ticks[boost_condition],
            )
            boost_mask[boost_condition] = 1.0
        else:
            boost_condition = np.zeros_like(stuck_candidate)

        applied_joint_delta = applied_ticks * RADIANS_PER_TICK
        telemetry = WrapperTelemetry(
            policy_joint_delta=policy_joint_delta.copy(),
            applied_joint_delta=applied_joint_delta.astype(np.float32),
            observed_joint_delta=observed_joint_delta.astype(np.float32),
            current_raw=current_raw.astype(np.float32),
            effective_min_ticks=effective_min_ticks.astype(np.float32),
            deadband_mask=deadband_mask.astype(np.float32),
            stuck_mask=stuck_mask.astype(np.float32),
            boost_mask=boost_mask.astype(np.float32),
        )
        self._last_wrapper_telemetry = telemetry

        if np.any(boost_condition):
            self._last_status_label = "stuck_boost"
        elif np.any(deadband_mask > 0.5):
            self._last_status_label = "deadband_comp"
        else:
            self._last_status_label = "tracking"
        return applied_joint_delta.astype(np.float32), telemetry

    def _compute_distance_scale(self, distance_to_goal: float, near_goal_scale: float) -> float:
        near_distance = min(self._cfg.wrapper_near_distance_m, self._cfg.wrapper_far_distance_m)
        far_distance = max(self._cfg.wrapper_near_distance_m, self._cfg.wrapper_far_distance_m)
        if far_distance <= near_distance + 1.0e-6:
            return float(near_goal_scale)
        clipped_distance = float(np.clip(distance_to_goal, near_distance, far_distance))
        alpha = (clipped_distance - near_distance) / (far_distance - near_distance)
        return float(near_goal_scale + alpha * (1.0 - near_goal_scale))

    def _build_observation(self) -> np.ndarray | None:
        ee_pos_w = self._lookup_ee_position()
        if ee_pos_w is None:
            return None

        joint_pos = self._state.joint_pos
        assert joint_pos is not None

        joint_pos_normalized = 2.0 * (joint_pos - self._cfg.joint_lower_limits) / (
            self._cfg.joint_upper_limits - self._cfg.joint_lower_limits + 1.0e-6
        ) - 1.0
        target_delta = self._cfg.target_pos - ee_pos_w
        wrist_roll_error = np.array([joint_pos[self._cfg.joint_names.index("Wrist_Roll")]], dtype=np.float32)
        jaw_error = np.array([joint_pos[self._cfg.joint_names.index("Jaw")]], dtype=np.float32)

        return np.concatenate(
            [
                joint_pos_normalized,
                target_delta,
                wrist_roll_error,
                jaw_error,
                self._prev_action,
            ]
        ).astype(np.float32)

    def _lookup_ee_position(self) -> np.ndarray | None:
        try:
            transform = self._tf_buffer.lookup_transform(
                self._cfg.tf_parent_frame,
                self._cfg.tf_child_frame,
                rclpy.time.Time(),
            )
        except TransformException:
            if not self._missing_tf_warned:
                self.get_logger().warning(
                    "No TF available yet for '%s' -> '%s'."
                    % (self._cfg.tf_parent_frame, self._cfg.tf_child_frame)
                )
                self._missing_tf_warned = True
            return None

        self._missing_tf_warned = False
        translation = transform.transform.translation
        ee_pos = np.array([translation.x, translation.y, translation.z], dtype=np.float32)
        if not self._is_finite_array(ee_pos, "ee position from tf"):
            return None
        return ee_pos

    def _publish_joint_target(self, joint_target: np.ndarray) -> None:
        if not self._is_finite_array(joint_target, "publish joint target"):
            return
        msg = JointState()
        msg.name = list(self._cfg.joint_names)
        msg.position = joint_target.tolist()
        self._command_pub.publish(msg)

    def _is_finite_array(self, value: np.ndarray, label: str) -> bool:
        if np.all(np.isfinite(value)):
            return True
        self.get_logger().error(f"Detected invalid {label}. Skipping this control step.")
        return False

    def _status_log_step(self) -> None:
        if self._state.joint_pos is None or self._last_distance_to_goal is None or self._last_ee_pos is None:
            self.get_logger().info("status=waiting joint_states/tf")
            return

        ee_x, ee_y, ee_z = self._last_ee_pos.tolist()
        target_preview = ", ".join(f"{value:.3f}" for value in self._joint_pos_target[:3].tolist())
        wrapper_summary = ""
        if self._cfg.enable_action_wrapper:
            boosted = int(np.sum(self._last_wrapper_telemetry.boost_mask))
            deadbanded = int(np.sum(self._last_wrapper_telemetry.deadband_mask))
            max_current = float(np.max(np.abs(self._last_wrapper_telemetry.current_raw)))
            wrapper_summary = f" deadband={deadbanded} boost={boosted} max_current={max_current:.1f}"
        self.get_logger().info(
            "status=%s distance=%.4fm ee=(%.3f, %.3f, %.3f) target_joints[:3]=[%s]%s"
            % (self._last_status_label, self._last_distance_to_goal, ee_x, ee_y, ee_z, target_preview, wrapper_summary)
        )

    def save_rollout(self) -> Path | None:
        if self._rollout_recorder is None:
            return None
        return self._rollout_recorder.save()


def _parse_float_list(value: str, expected_len: int, label: str) -> np.ndarray:
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(f"{label} must contain exactly {expected_len} comma-separated values.")
    return np.asarray(parts, dtype=np.float32)


def parse_args() -> RunnerConfig:
    parser = argparse.ArgumentParser(description="Run the trained so101 policy against the real robot.")
    parser.add_argument("--policy-path", type=Path, required=True, help="Path to policy.onnx or policy.pt.")
    parser.add_argument(
        "--target-pos",
        type=lambda value: _parse_float_list(value, 3, "target-pos"),
        default=DEFAULT_TARGET_POS.copy(),
        help="Fixed target position as x,y,z in the tf_parent_frame.",
    )
    parser.add_argument("--joint-state-topic", default="/joint_states", help="Input JointState topic from the robot.")
    parser.add_argument(
        "--current-topic",
        default="/follower/joint_currents",
        help="JointState topic carrying per-joint current-like feedback.",
    )
    parser.add_argument(
        "--current-state-field",
        default="effort",
        choices=("position", "velocity", "effort"),
        help="Field used to read current-like values from the current-topic JointState.",
    )
    parser.add_argument(
        "--command-topic",
        default="/so101/policy_joint_command",
        help="Output JointState topic for commanded joint targets.",
    )
    parser.add_argument("--tf-parent-frame", default="base_link", help="Parent TF frame used for EE lookup.")
    parser.add_argument("--tf-child-frame", default="gripper_frame_link", help="Child TF frame used for EE lookup.")
    parser.add_argument("--control-period", type=float, default=1.0 / 60.0, help="Policy execution period in seconds.")
    parser.add_argument(
        "--action-scale-gain",
        type=float,
        default=DEFAULT_ACTION_SCALE_GAIN,
        help="Gain applied to action deltas.",
    )
    parser.add_argument(
        "--max-delta-scale",
        type=float,
        default=0.5,
        help="Additional safety clamp relative to the learned action scale.",
    )
    parser.add_argument(
        "--stale-timeout-s",
        type=float,
        default=0.2,
        help="Skip policy steps if joint states stop updating for longer than this timeout.",
    )
    parser.add_argument(
        "--current-stale-timeout-s",
        type=float,
        default=0.2,
        help="Treat current feedback as stale after this many seconds.",
    )
    parser.add_argument(
        "--status-log-period-s",
        type=float,
        default=0.5,
        help="Periodic status log interval in seconds.",
    )
    parser.add_argument(
        "--enable-action-wrapper",
        action="store_true",
        default=False,
        help="Enable deadband compensation and stuck-aware boost for hardware execution.",
    )
    parser.add_argument(
        "--disable-deadband-compensation",
        action="store_true",
        default=False,
        help="Disable the minimum effective step compensation inside the hardware wrapper.",
    )
    parser.add_argument(
        "--disable-stuck-boost",
        action="store_true",
        default=False,
        help="Disable current-aware boost when the joint appears stalled.",
    )
    parser.add_argument(
        "--wrapper-far-distance-m",
        type=float,
        default=0.05,
        help="Distance above which the full deadband and boost tick settings are used.",
    )
    parser.add_argument(
        "--wrapper-near-distance-m",
        type=float,
        default=0.01,
        help="Distance below which wrapper tick settings are scaled down for fine alignment.",
    )
    parser.add_argument(
        "--near-goal-tick-scale",
        type=float,
        default=0.35,
        help="Scale applied to stuck-boost tick settings near the goal distance threshold.",
    )
    parser.add_argument(
        "--deadband-near-goal-tick-scale",
        type=float,
        default=0.10,
        help="Scale applied to deadband compensation tick settings near the goal distance threshold.",
    )
    parser.add_argument(
        "--min-effective-ticks",
        type=lambda value: _parse_float_list(value, len(CONTROLLED_JOINTS), "min-effective-ticks"),
        default=DEFAULT_MIN_EFFECTIVE_TICKS.copy(),
        help="Per-joint minimum effective step in ticks for deadband compensation.",
    )
    parser.add_argument(
        "--deadband-enable-mask",
        type=lambda value: _parse_float_list(value, len(CONTROLLED_JOINTS), "deadband-enable-mask"),
        default=DEFAULT_DEADBAND_ENABLE_MASK.copy(),
        help="Per-joint deadband enable mask. Use 1 for enabled and 0 for disabled.",
    )
    parser.add_argument(
        "--deadband-persistence-steps",
        type=int,
        default=3,
        help="Number of consecutive same-direction small commands before deadband compensation is applied.",
    )
    parser.add_argument(
        "--motion-epsilon-ticks",
        type=lambda value: _parse_float_list(value, len(CONTROLLED_JOINTS), "motion-epsilon-ticks"),
        default=DEFAULT_MOTION_EPSILON_TICKS.copy(),
        help="Per-joint observed motion threshold in ticks below which the joint is treated as stationary.",
    )
    parser.add_argument(
        "--stuck-current-thresholds",
        type=lambda value: _parse_float_list(value, len(CONTROLLED_JOINTS), "stuck-current-thresholds"),
        default=DEFAULT_STUCK_CURRENT_THRESHOLDS.copy(),
        help="Per-joint current_raw thresholds used to detect stalled high-load behavior.",
    )
    parser.add_argument(
        "--stuck-boost-ticks",
        type=lambda value: _parse_float_list(value, len(CONTROLLED_JOINTS), "stuck-boost-ticks"),
        default=DEFAULT_STUCK_BOOST_TICKS.copy(),
        help="Per-joint minimum command size in ticks applied once a joint is classified as stuck.",
    )
    parser.add_argument(
        "--stuck-consecutive-steps",
        type=int,
        default=3,
        help="Number of consecutive stalled steps required before boost is activated.",
    )
    parser.add_argument(
        "--record-rollout",
        action="store_true",
        default=False,
        help="Record real rollout data for later analysis/fine-tuning.",
    )
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=Path("logs/real_rollouts"),
        help="Directory where recorded real rollouts are saved.",
    )
    parser.add_argument(
        "--rollout-name",
        type=str,
        default=None,
        help="Optional rollout file stem. A timestamped name is used when omitted.",
    )
    args = parser.parse_args()

    return RunnerConfig(
        policy_path=args.policy_path,
        target_pos=args.target_pos,
        joint_state_topic=args.joint_state_topic,
        current_topic=args.current_topic,
        current_state_field=args.current_state_field,
        command_topic=args.command_topic,
        tf_parent_frame=args.tf_parent_frame,
        tf_child_frame=args.tf_child_frame,
        control_period=args.control_period,
        action_scale_gain=args.action_scale_gain,
        max_delta_scale=args.max_delta_scale,
        stale_timeout_s=args.stale_timeout_s,
        current_stale_timeout_s=args.current_stale_timeout_s,
        status_log_period_s=args.status_log_period_s,
        enable_action_wrapper=args.enable_action_wrapper,
        enable_deadband_compensation=not args.disable_deadband_compensation,
        enable_stuck_boost=not args.disable_stuck_boost,
        wrapper_far_distance_m=args.wrapper_far_distance_m,
        wrapper_near_distance_m=args.wrapper_near_distance_m,
        near_goal_tick_scale=args.near_goal_tick_scale,
        deadband_near_goal_tick_scale=args.deadband_near_goal_tick_scale,
        min_effective_ticks=args.min_effective_ticks,
        deadband_enable_mask=args.deadband_enable_mask,
        deadband_persistence_steps=args.deadband_persistence_steps,
        motion_epsilon_ticks=args.motion_epsilon_ticks,
        stuck_current_thresholds=args.stuck_current_thresholds,
        stuck_boost_ticks=args.stuck_boost_ticks,
        stuck_consecutive_steps=args.stuck_consecutive_steps,
        record_rollout=args.record_rollout,
        rollout_dir=args.rollout_dir,
        rollout_name=args.rollout_name,
    )


def main() -> None:
    config = parse_args()
    rclpy.init()
    node = So101RealPolicyRunner(config)
    try:
        rclpy.spin(node)
    finally:
        saved_rollout = node.save_rollout()
        if saved_rollout is not None:
            node.get_logger().info(f"Saved rollout to: {saved_rollout}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
