import hydra
import os
from pathlib import Path
import torch
from tqdm import trange

from omni_drones import init_simulation_app
from tensordict import TensorDict


@hydra.main(config_path=os.path.dirname(__file__), config_name=Path(__file__).stem)
def main(cfg):
    app = init_simulation_app(cfg)

    # due to the design of Isaac Sim, these imports are only available
    # after the SimulationApp instance is created
    from omni_drones.envs.isaac_env import IsaacEnv
    from omni_drones.robots.assets import Multirotor, get_robot_cfg
    from omni_drones.utils.torch import euler_to_quaternion

    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.terrains import TerrainImporterCfg
    import isaaclab.sim as sim_utils

    class MyEnv(IsaacEnv):
        def __init__(self, cfg):
            super().__init__(cfg)
            # the `__init__` method invokes `_design_scene` to create the scene
            # after that, all the entities created are managed by `self.scene`
            print(self.scene)

            # let's get the drone entity
            self.drone: Multirotor = self.scene["drone"]
            self.default_init_state = self.drone.data.default_root_state.clone()
            default_init_pos = self.drone.data.root_pos_w

            self.target_pos = torch.zeros(default_init_pos.size(), device=self.device)
            self.target_yaw = torch.zeros(self.drone.shape, device=self.device)

            self.resolve_specs()

        def _design_scene(self):
            # the scene is created from a SceneCfg object in a declarative way
            # see the docstring of `InteractiveSceneCfg` for more details
            class SceneCfg(InteractiveSceneCfg):
                terrain = TerrainImporterCfg(
                    prim_path="/World/ground",
                    terrain_type="plane",
                    collision_group=-1,
                )
                # lights
                light = AssetBaseCfg(
                    prim_path="/World/light",
                    spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
                )
                sky_light = AssetBaseCfg(
                    prim_path="/World/skyLight",
                    spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
                )

                drone = get_robot_cfg(cfg.robot_name).replace(
                    prim_path="{ENV_REGEX_NS}/Robot",
                )

            return SceneCfg(num_envs=cfg.num_envs, env_spacing=cfg.env_spacing)

        def _reset_idx(self, env_ids: torch.Tensor):
            # since we have multiple parallel environments
            # the environment offset is added to the initial state
            init_state = self.default_init_state[env_ids]
            init_state[:, :3] += self.scene.env_origins[env_ids]

            init_rpy = torch.zeros(self.scene.env_origins.size(), device=self.device)
            init_rpy[0, 2] = 45.0 / 180.0 * torch.pi
            init_rpy[1, 2] = 135.0 / 180.0 * torch.pi
            init_rpy[3, 2] = 225.0 / 180.0 * torch.pi
            init_rpy[2, 2] = 315.0 / 180.0 * torch.pi
            init_state[:, 3:7] = euler_to_quaternion(init_rpy)

            self.drone.write_root_state_to_sim(init_state, env_ids)

    env = MyEnv(cfg)

    radius = (torch.ones(1, device=env.device) * 2.5).sqrt()
    omega = torch.ones(1, device=env.device) * 1.
    phase = torch.as_tensor([0., 1., 3., 2.], device=env.device) / 4.
    def compute_ref(t):
        _t = -torch.pi/4. + phase * 2*torch.pi + t * omega
        pos = torch.stack([
            torch.cos(_t) * radius,
            torch.sin(_t) * radius,
            torch.ones(env.num_envs, device=env.device) * 1.5
        ], dim=-1)
        pos -= env.scene.env_origins

        vel_xy = torch.stack([
            -torch.sin(_t) * radius * omega,
            torch.cos(_t) * radius * omega,
        ], dim=-1)
        yaw = torch.atan2(vel_xy[:, 1], vel_xy[:, 0])
        return pos, yaw

    def policy(cur_t: float, tensordict: TensorDict):
        ref_pos, ref_yaw = compute_ref(cur_t)
        target_pos = ref_pos
        target_yaw = ref_yaw.reshape(-1, 1)
        action = torch.cat([target_pos, target_yaw], dim=-1)
        tensordict["agents", "action"] = action
        # print(f"{tensordict['agents', 'action']=}")
        return tensordict

    def rollout(env: IsaacEnv):
        data_ = env.reset()
        result = []
        for i in trange(env.max_episode_length):
            cur_t = (i % env.max_episode_length) * cfg.sim.dt
            data_ = policy(cur_t, data_)
            data, data_ = env.step_and_maybe_reset(data_)
            result.append(data)
        return torch.stack(result)

    while app.is_running():
        rollout(env)
    app.close()


if __name__ == "__main__":
    main()
