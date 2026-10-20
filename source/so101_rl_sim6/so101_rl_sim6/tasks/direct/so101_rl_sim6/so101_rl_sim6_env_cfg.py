# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg


USD_PATH = Path(__file__).resolve().parents[6] / "so101_3.usd"


@configclass
class So101RlSim6EnvCfg(DirectRLEnvCfg):
    """SO-101 reach task migrated to Isaac Lab 3 / Isaac Sim 6."""

    # Environment contract preserved from the model_399 training run.
    decimation = 1
    episode_length_s = 60.0
    action_space = 6
    observation_space = 17
    state_space = 0

    # Keep the original 60 Hz PhysX control and simulation rate.
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics=PhysxCfg(),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "Rotation": 0.0,
                "Pitch": 0.0,
                "Elbow": 0.0,
                "Wrist_Pitch": 0.0,
                "Wrist_Roll": 0.0,
                "Jaw": 0.0,
            },
        ),
        actuators={
            "so101_arm": ImplicitActuatorCfg(
                joint_names_expr=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
                stiffness=180.0,
                damping=24.0,
            ),
        },
    )

    controlled_joints = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    ee_body_name = "gripper_frame_link"

    # Incremental position-control action contract.
    action_scale = (0.08, 0.08, 0.08, 0.10, 0.13, 0.06)
    action_scale_gain = 0.042

    # Reset and target configuration.
    joint_random_ratio = 0.65
    jaw_reset_bias_to_closed = 0.70
    target_pos = (0.180, 0.000, 0.200)

    # Success thresholds and bonuses.
    success_thresholds_m = (0.020, 0.010, 0.005, 0.003, 0.001)
    success_bonus_20mm = 2.0
    success_bonus_10mm = 4.0
    success_bonus_5mm = 8.0
    success_bonus_3mm = 12.0
    success_bonus_1mm = 20.0
    hold_bonus_1mm = 4.0
    # Awarded once after the strict 1 mm + zero-pose condition is held for
    # ``success_hold_steps``. With gamma=0.99, the upper discounted value of
    # lingering inside the precision region under the shaping terms below is
    # about 3.8k, so the completion bonus must be larger than that value.
    success_completion_bonus = 5000.0

    # Dense rewards and penalties.
    reward_distance_exp_scale = 30.0
    reward_progress_scale = 36.0
    reward_goal_tracking_scale = 4.0
    reward_time_penalty = -0.01
    reward_action_rate_penalty_scale = -0.05
    reward_joint_velocity_penalty_scale = -0.002
    reward_stall_penalty = -0.15
    reward_ee_still_scale = 3.0
    reward_joint_still_scale = 2.0
    reward_action_smooth_scale = 1.5
    reward_wrist_roll_zero_scale = 2.0
    reward_jaw_zero_scale = 1.0
    reward_wrist_roll_abs_penalty_scale = -0.25
    reward_jaw_abs_penalty_scale = -0.25
    reward_zero_pose_far_scale = 0.20
    reward_zero_pose_near_scale = 1.0
    reward_zero_pose_precision_scale = 8.0

    stall_distance_m = 0.05
    stall_speed_mps = 0.01
    near_goal_distance_m = 0.02
    precision_goal_distance_m = 0.003
    success_wrist_roll_threshold_rad = 0.05
    success_jaw_threshold_rad = 0.05
    success_hold_steps = 12
