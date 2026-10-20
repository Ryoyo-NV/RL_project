# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .so101_rl_sim6_env_cfg import So101RlSim6EnvCfg


class So101RlSim6Env(DirectRLEnv):
    """Direct SO-101 reaching environment for Isaac Sim 6."""

    cfg: So101RlSim6EnvCfg

    def __init__(self, cfg: So101RlSim6EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_ids, joint_names = self.robot.find_joints(
            self.cfg.controlled_joints,
            preserve_order=True,
        )
        if joint_names != list(self.cfg.controlled_joints):
            raise RuntimeError(
                "SO-101 joint order does not match the policy contract. "
                f"Expected {self.cfg.controlled_joints}, resolved {joint_names}."
            )

        ee_body_indices, ee_body_names = self.robot.find_bodies(self.cfg.ee_body_name, preserve_order=True)
        if len(ee_body_indices) != 1:
            raise RuntimeError(
                f"Expected one end-effector body named '{self.cfg.ee_body_name}', "
                f"resolved {ee_body_names} at indices {ee_body_indices}."
            )
        self._ee_body_idx = ee_body_indices[0]

        joint_limits = self.robot.data.soft_joint_pos_limits.torch[0, self._joint_ids]
        self._joint_lower_limits = joint_limits[:, 0].clone()
        self._joint_upper_limits = joint_limits[:, 1].clone()
        self._default_joint_pos = self.robot.data.default_joint_pos.torch[:, self._joint_ids].clone()
        self._default_joint_vel = self.robot.data.default_joint_vel.torch[:, self._joint_ids].clone()

        self._jaw_joint_local_idx = self.cfg.controlled_joints.index("Jaw")
        self._wrist_roll_joint_local_idx = self.cfg.controlled_joints.index("Wrist_Roll")
        self._action_scale = torch.tensor(
            self.cfg.action_scale,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        self._action_scale_gain = float(self.cfg.action_scale_gain)
        self._target_pos = torch.tensor(self.cfg.target_pos, device=self.device, dtype=torch.float32)
        if len(self.cfg.success_thresholds_m) != 5:
            raise ValueError("SO-101 reward logic requires exactly five success thresholds: 20/10/5/3/1 mm.")
        self._threshold_bonuses = torch.tensor(
            (
                self.cfg.success_bonus_20mm,
                self.cfg.success_bonus_10mm,
                self.cfg.success_bonus_5mm,
                self.cfg.success_bonus_3mm,
                self.cfg.success_bonus_1mm,
            ),
            device=self.device,
            dtype=torch.float32,
        )

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.joint_pos_target = self._default_joint_pos.clone()
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_distance = torch.zeros(self.num_envs, device=self.device)
        self.success_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._episode_distance_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_step_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_min_distance = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._episode_ee_speed_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_wrist_roll_abs_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_jaw_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_jaw_abs_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_jaw_action_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_jaw_action_abs_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_near_goal_joint_speed_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_near_goal_step_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_reward_jaw_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_reward_threshold_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_reward_completion_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_reward_total_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_hit_20mm = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_hit_10mm = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_hit_5mm = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_hit_3mm = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_hit_1mm = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_hit_1mm_zero_pose = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_completed_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reward_threshold_claimed = torch.zeros(
            (self.num_envs, len(self.cfg.success_thresholds_m)),
            dtype=torch.bool,
            device=self.device,
        )

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)
        joint_delta = self.actions * self._action_scale * self._action_scale_gain
        self.joint_pos_target = torch.clamp(
            self.joint_pos_target + joint_delta,
            self._joint_lower_limits.unsqueeze(0),
            self._joint_upper_limits.unsqueeze(0),
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target_index(
            target=self.joint_pos_target,
            joint_ids=self._joint_ids,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        joint_pos = self.robot.data.joint_pos.torch[:, self._joint_ids]
        ee_pos_w = self.robot.data.body_link_pos_w.torch[:, self._ee_body_idx]

        joint_pos_normalized = 2.0 * (joint_pos - self._joint_lower_limits) / (
            self._joint_upper_limits - self._joint_lower_limits + 1.0e-6
        ) - 1.0
        target_delta = self.goal_pos_w - ee_pos_w
        wrist_roll_error = joint_pos[:, self._wrist_roll_joint_local_idx].unsqueeze(-1)
        jaw_error = joint_pos[:, self._jaw_joint_local_idx].unsqueeze(-1)

        obs = torch.cat(
            (
                joint_pos_normalized,
                target_delta,
                wrist_roll_error,
                jaw_error,
                self.prev_actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos.torch[:, self._joint_ids]
        joint_vel = self.robot.data.joint_vel.torch[:, self._joint_ids]
        ee_pos_w = self.robot.data.body_link_pos_w.torch[:, self._ee_body_idx]
        ee_lin_vel_w = self.robot.data.body_com_lin_vel_w.torch[:, self._ee_body_idx]

        distance = torch.linalg.norm(self.goal_pos_w - ee_pos_w, dim=-1)
        progress_reward = (self.prev_distance - distance) * self.cfg.reward_progress_scale
        self.prev_distance = distance.clone()

        dense_goal_reward = (
            torch.exp(-self.cfg.reward_distance_exp_scale * distance) * self.cfg.reward_goal_tracking_scale
        )

        wrist_roll_error = torch.abs(joint_pos[:, self._wrist_roll_joint_local_idx])
        jaw_pos = joint_pos[:, self._jaw_joint_local_idx]
        jaw_zero_error = torch.abs(jaw_pos)
        zero_pose_ok = (
            (wrist_roll_error <= self.cfg.success_wrist_roll_threshold_rad)
            & (jaw_zero_error <= self.cfg.success_jaw_threshold_rad)
        )

        # Milestone bonuses are one-shot rewards. Repeating them every step
        # allowed the policy to farm the 20/10/5 mm bonuses for the rest of a
        # 60 second episode without satisfying the strict pose condition.
        threshold_conditions = torch.stack(
            (
                distance <= self.cfg.success_thresholds_m[0],
                distance <= self.cfg.success_thresholds_m[1],
                distance <= self.cfg.success_thresholds_m[2],
                (distance <= self.cfg.success_thresholds_m[3]) & zero_pose_ok,
                (distance <= self.cfg.success_thresholds_m[4]) & zero_pose_ok,
            ),
            dim=-1,
        )
        new_threshold_hits = threshold_conditions & ~self._reward_threshold_claimed
        threshold_reward = torch.sum(
            new_threshold_hits.float() * self._threshold_bonuses.unsqueeze(0),
            dim=-1,
        )
        self._reward_threshold_claimed |= threshold_conditions

        within_best = (distance <= self.cfg.success_thresholds_m[4]) & zero_pose_ok
        hold_reward = within_best.float() * self.cfg.hold_bonus_1mm
        # DirectRLEnv calls _get_dones() before _get_rewards(), so the success
        # counter for this state has already been updated.
        completion_reached = (
            (self.success_counter == self.cfg.success_hold_steps)
            & ~self._episode_completed_success
        )
        completion_reward = completion_reached.float() * self.cfg.success_completion_bonus
        self._episode_completed_success |= completion_reached

        ee_speed = torch.linalg.norm(ee_lin_vel_w, dim=-1)
        stall_mask = (distance > self.cfg.stall_distance_m) & (ee_speed < self.cfg.stall_speed_mps)
        stall_penalty = stall_mask.float() * self.cfg.reward_stall_penalty

        action_rate_penalty = torch.sum(torch.square(self.actions - self.prev_actions), dim=-1)
        joint_velocity_penalty = torch.sum(torch.square(joint_vel), dim=-1)
        near_goal_weight = torch.clamp(
            1.0 - distance / self.cfg.near_goal_distance_m,
            min=0.0,
            max=1.0,
        )
        precision_goal_weight = (distance <= self.cfg.precision_goal_distance_m).float()
        stillness_weight = torch.maximum(near_goal_weight, precision_goal_weight)
        zero_pose_weight = torch.full_like(distance, self.cfg.reward_zero_pose_far_scale)
        zero_pose_weight += near_goal_weight * self.cfg.reward_zero_pose_near_scale
        zero_pose_weight += precision_goal_weight * self.cfg.reward_zero_pose_precision_scale

        wrist_roll_zero_reward = (
            torch.exp(-8.0 * wrist_roll_error)
            * self.cfg.reward_wrist_roll_zero_scale
            * zero_pose_weight
        )
        # The task's success target is Jaw == 0. Use a symmetric error so both
        # positive and negative deviations receive a restoring signal.
        jaw_zero_reward = (
            torch.exp(-10.0 * jaw_zero_error)
            * self.cfg.reward_jaw_zero_scale
            * zero_pose_weight
        )
        wrist_roll_abs_penalty = wrist_roll_error * self.cfg.reward_wrist_roll_abs_penalty_scale
        jaw_abs_penalty = jaw_zero_error * self.cfg.reward_jaw_abs_penalty_scale

        ee_still_reward = torch.exp(-40.0 * ee_speed) * self.cfg.reward_ee_still_scale * stillness_weight
        joint_still_reward = (
            torch.exp(-2.5 * torch.linalg.norm(joint_vel, dim=-1))
            * self.cfg.reward_joint_still_scale
            * stillness_weight
        )
        action_smooth_reward = (
            torch.exp(-8.0 * torch.linalg.norm(self.actions - self.prev_actions, dim=-1))
            * self.cfg.reward_action_smooth_scale
            * stillness_weight
        )

        reward = dense_goal_reward
        reward += progress_reward
        reward += wrist_roll_zero_reward
        reward += jaw_zero_reward
        reward += wrist_roll_abs_penalty
        reward += jaw_abs_penalty
        reward += threshold_reward
        reward += hold_reward
        reward += completion_reward
        reward += ee_still_reward
        reward += joint_still_reward
        reward += action_smooth_reward
        reward += stall_penalty
        reward += self.cfg.reward_time_penalty
        reward += action_rate_penalty * self.cfg.reward_action_rate_penalty_scale
        reward += joint_velocity_penalty * self.cfg.reward_joint_velocity_penalty_scale

        self._episode_reward_jaw_sum += jaw_zero_reward + jaw_abs_penalty
        self._episode_reward_threshold_sum += threshold_reward
        self._episode_reward_completion_sum += completion_reward
        self._episode_reward_total_sum += reward
        self._update_logging_metrics(distance, ee_speed, joint_pos, joint_vel)
        self.prev_actions = self.actions.clone()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        joint_pos = self.robot.data.joint_pos.torch[:, self._joint_ids]
        ee_pos_w = self.robot.data.body_link_pos_w.torch[:, self._ee_body_idx]
        distance = torch.linalg.norm(self.goal_pos_w - ee_pos_w, dim=-1)

        episode_limit_reached = self.episode_length_buf >= self.max_episode_length - 1
        invalid_state = torch.any(torch.isnan(joint_pos), dim=-1) | torch.any(torch.isnan(ee_pos_w), dim=-1)
        far_away = distance > 1.0

        wrist_roll_error = torch.abs(joint_pos[:, self._wrist_roll_joint_local_idx])
        jaw_zero_error = torch.abs(joint_pos[:, self._jaw_joint_local_idx])
        within_best = (
            (distance <= self.cfg.success_thresholds_m[4])
            & (wrist_roll_error <= self.cfg.success_wrist_roll_threshold_rad)
            & (jaw_zero_error <= self.cfg.success_jaw_threshold_rad)
        )
        self.success_counter = torch.where(
            within_best,
            self.success_counter + 1,
            torch.zeros_like(self.success_counter),
        )
        success_done = self.success_counter >= self.cfg.success_hold_steps
        terminated = invalid_state | far_away | success_done
        # Do not let RSL-RL bootstrap a terminal success if it happens on the
        # same step as the episode limit.
        timed_out = episode_limit_reached & ~terminated
        return terminated, timed_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        # Initial reset has no completed episode. Excluding it avoids publishing
        # placeholder zeros and an infinite minimum distance to the RL logger.
        completed_env_ids = env_ids[self._episode_step_count[env_ids] > 0]
        if completed_env_ids.numel() > 0:
            episode_length = self._episode_step_count[completed_env_ids]
            near_goal_steps = torch.clamp(
                self._episode_near_goal_step_count[completed_env_ids],
                min=1.0,
            )
            final_joint_pos = self.robot.data.joint_pos.torch[completed_env_ids][:, self._joint_ids]
            final_jaw_pos = final_joint_pos[:, self._jaw_joint_local_idx]
            self.extras["log"] = {
                "Metrics/episode_length_mean": torch.mean(episode_length).item(),
                "Metrics/distance_mean": torch.mean(
                    self._episode_distance_sum[completed_env_ids] / episode_length
                ).item(),
                "Metrics/distance_min": torch.mean(
                    self._episode_min_distance[completed_env_ids]
                ).item(),
                "Metrics/success_rate_20mm": self._episode_hit_20mm[
                    completed_env_ids
                ].float().mean().item(),
                "Metrics/success_rate_10mm": self._episode_hit_10mm[
                    completed_env_ids
                ].float().mean().item(),
                "Metrics/success_rate_5mm": self._episode_hit_5mm[
                    completed_env_ids
                ].float().mean().item(),
                "Metrics/success_rate_3mm": self._episode_hit_3mm[
                    completed_env_ids
                ].float().mean().item(),
                "Metrics/success_rate_1mm": self._episode_hit_1mm[
                    completed_env_ids
                ].float().mean().item(),
                "Metrics/success_rate_1mm_zero_pose": self._episode_hit_1mm_zero_pose[
                    completed_env_ids
                ].float().mean().item(),
                "Metrics/wrist_roll_abs_mean": torch.mean(
                    self._episode_wrist_roll_abs_sum[completed_env_ids] / episode_length
                ).item(),
                "Metrics/jaw_abs_mean": torch.mean(
                    self._episode_jaw_abs_sum[completed_env_ids] / episode_length
                ).item(),
                "Metrics/jaw_mean": torch.mean(
                    self._episode_jaw_sum[completed_env_ids] / episode_length
                ).item(),
                "Metrics/jaw_final_mean": torch.mean(final_jaw_pos).item(),
                "Metrics/jaw_final_abs_mean": torch.mean(torch.abs(final_jaw_pos)).item(),
                "Metrics/jaw_action_mean": torch.mean(
                    self._episode_jaw_action_sum[completed_env_ids] / episode_length
                ).item(),
                "Metrics/jaw_action_abs_mean": torch.mean(
                    self._episode_jaw_action_abs_sum[completed_env_ids] / episode_length
                ).item(),
                "Metrics/strict_success_rate": self._episode_completed_success[
                    completed_env_ids
                ].float().mean().item(),
                "Metrics/ee_speed_mean": torch.mean(
                    self._episode_ee_speed_sum[completed_env_ids] / episode_length
                ).item(),
                "Metrics/near_goal_joint_speed_mean": torch.mean(
                    self._episode_near_goal_joint_speed_sum[completed_env_ids] / near_goal_steps
                ).item(),
                "RewardComponents/jaw_mean": torch.mean(
                    self._episode_reward_jaw_sum[completed_env_ids] / episode_length
                ).item(),
                "RewardComponents/threshold_sum": torch.mean(
                    self._episode_reward_threshold_sum[completed_env_ids]
                ).item(),
                "RewardComponents/completion_sum": torch.mean(
                    self._episode_reward_completion_sum[completed_env_ids]
                ).item(),
                "RewardComponents/total_mean": torch.mean(
                    self._episode_reward_total_sum[completed_env_ids] / episode_length
                ).item(),
            }

        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()

        joint_span = self._joint_upper_limits - self._joint_lower_limits
        randomized_joint_pos = self._default_joint_pos[env_ids] + (
            torch.rand((env_ids.numel(), len(self._joint_ids)), device=self.device) - 0.5
        ) * joint_span.unsqueeze(0) * self.cfg.joint_random_ratio
        randomized_joint_pos = torch.clamp(
            randomized_joint_pos,
            self._joint_lower_limits.unsqueeze(0),
            self._joint_upper_limits.unsqueeze(0),
        )

        jaw_low = self._joint_lower_limits[self._jaw_joint_local_idx]
        jaw_high = self._joint_upper_limits[self._jaw_joint_local_idx]
        jaw_range = jaw_high - jaw_low
        randomized_joint_pos[:, self._jaw_joint_local_idx] = sample_uniform(
            jaw_low.item(),
            (jaw_low + jaw_range * self.cfg.jaw_reset_bias_to_closed).item(),
            (env_ids.numel(),),
            self.device,
        )

        joint_pos[:, self._joint_ids] = randomized_joint_pos
        joint_vel[:, self._joint_ids] = 0.0

        default_root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        default_root_vel = self.robot.data.default_root_vel.torch[env_ids].clone()

        self.robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        self.joint_pos_target[env_ids] = randomized_joint_pos
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.success_counter[env_ids] = 0

        self.goal_pos_w[env_ids] = self._target_pos.unsqueeze(0).repeat(env_ids.numel(), 1)
        self.goal_pos_w[env_ids] += self.scene.env_origins[env_ids]

        ee_pos_w = self.robot.data.body_link_pos_w.torch[env_ids, self._ee_body_idx]
        self.prev_distance[env_ids] = torch.linalg.norm(self.goal_pos_w[env_ids] - ee_pos_w, dim=-1)
        self._episode_distance_sum[env_ids] = 0.0
        self._episode_step_count[env_ids] = 0.0
        self._episode_min_distance[env_ids] = float("inf")
        self._episode_ee_speed_sum[env_ids] = 0.0
        self._episode_wrist_roll_abs_sum[env_ids] = 0.0
        self._episode_jaw_sum[env_ids] = 0.0
        self._episode_jaw_abs_sum[env_ids] = 0.0
        self._episode_jaw_action_sum[env_ids] = 0.0
        self._episode_jaw_action_abs_sum[env_ids] = 0.0
        self._episode_near_goal_joint_speed_sum[env_ids] = 0.0
        self._episode_near_goal_step_count[env_ids] = 0.0
        self._episode_reward_jaw_sum[env_ids] = 0.0
        self._episode_reward_threshold_sum[env_ids] = 0.0
        self._episode_reward_completion_sum[env_ids] = 0.0
        self._episode_reward_total_sum[env_ids] = 0.0
        self._episode_hit_20mm[env_ids] = False
        self._episode_hit_10mm[env_ids] = False
        self._episode_hit_5mm[env_ids] = False
        self._episode_hit_3mm[env_ids] = False
        self._episode_hit_1mm[env_ids] = False
        self._episode_hit_1mm_zero_pose[env_ids] = False
        self._episode_completed_success[env_ids] = False
        self._reward_threshold_claimed[env_ids] = False

    def _update_logging_metrics(
        self,
        distance: torch.Tensor,
        ee_speed: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> None:
        self._episode_distance_sum += distance
        self._episode_step_count += 1.0
        self._episode_min_distance = torch.minimum(self._episode_min_distance, distance)
        self._episode_ee_speed_sum += ee_speed
        self._episode_wrist_roll_abs_sum += torch.abs(joint_pos[:, self._wrist_roll_joint_local_idx])
        jaw_pos = joint_pos[:, self._jaw_joint_local_idx]
        jaw_action = self.actions[:, self._jaw_joint_local_idx]
        self._episode_jaw_sum += jaw_pos
        self._episode_jaw_abs_sum += torch.abs(jaw_pos)
        self._episode_jaw_action_sum += jaw_action
        self._episode_jaw_action_abs_sum += torch.abs(jaw_action)

        joint_speed = torch.linalg.norm(joint_vel, dim=-1)
        near_goal_mask = distance <= self.cfg.near_goal_distance_m
        self._episode_near_goal_joint_speed_sum += joint_speed * near_goal_mask.float()
        self._episode_near_goal_step_count += near_goal_mask.float()

        self._episode_hit_20mm |= distance <= self.cfg.success_thresholds_m[0]
        self._episode_hit_10mm |= distance <= self.cfg.success_thresholds_m[1]
        self._episode_hit_5mm |= distance <= self.cfg.success_thresholds_m[2]
        self._episode_hit_3mm |= distance <= self.cfg.success_thresholds_m[3]
        self._episode_hit_1mm |= distance <= self.cfg.success_thresholds_m[4]
        self._episode_hit_1mm_zero_pose |= (
            (distance <= self.cfg.success_thresholds_m[4])
            & (
                torch.abs(joint_pos[:, self._wrist_roll_joint_local_idx])
                <= self.cfg.success_wrist_roll_threshold_rad
            )
            & (torch.abs(joint_pos[:, self._jaw_joint_local_idx]) <= self.cfg.success_jaw_threshold_rad)
        )
