from omni_drones.robots.multirotor import Multirotor, RotorCfg
from omni_drones.utils.lab import DEFAULT_CFG
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

import os.path as osp

ASSET_PATH = osp.join(osp.dirname(__file__), "assets")

HUMMINGBIRD_CFG = ArticulationCfg(
    class_type=Multirotor,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/usd/hummingbird.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),
    ),
    actuators={
        "rotor": RotorCfg(
            joint_names_expr=None,
            stiffness=None,
            damping=None,
            body_names_expr=["rotor_.*"],
            max_rotor_speed=838,
            kf=8.54858e-06,
            km=1.3677728816219314e-07,
            rotor_direction={
                "rotor_(0|2)": -1.0,
                "rotor_(1|3)": 1.0,
            },
            tau_up=0.43,
            tau_down=0.43,
        )
    }
)


FIREFLY_CFG = ArticulationCfg(
    class_type=Multirotor,
    spawn=DEFAULT_CFG.replace(usd_path=f"{ASSET_PATH}/usd/firefly.usd"),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),
    ),
    actuators={
        "rotor": RotorCfg(
            joint_names_expr=None,
            stiffness=None,
            damping=None,
            body_names_expr=["rotor_.*"],
            max_rotor_speed=838,
            kf=8.54858e-06,
            km=1.3677728816219314e-07,
            rotor_direction={
                "rotor_(0|2|4)": 1.0,
                "rotor_(1|3|5)": -1.0,
            },
            tau_up=0.5,
            tau_down=0.5,
        )
    }
)

def get_robot_cfg(robot_name: str) -> ArticulationCfg:
    if robot_name == "hummingbird":
        return HUMMINGBIRD_CFG
    elif robot_name == "firefly":
        return FIREFLY_CFG
    else:
        raise ValueError(f"Unknown robot name: {robot_name}")
