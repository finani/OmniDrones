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

            self.target_roll = torch.zeros(self.drone.shape, device=self.device)
            self.target_roll[:] = cfg.goal[0] / 180.0 * torch.pi
            self.target_pitch = torch.zeros(self.drone.shape, device=self.device)
            self.target_pitch[:] = cfg.goal[1] / 180.0 * torch.pi
            self.target_yaw = torch.zeros(self.drone.shape, device=self.device)
            self.target_yaw[:] = cfg.goal[2] / 180.0 * torch.pi

            # self.drone_mass = self.drone.root_physx_view.get_masses().sum(-1, keepdim=True).to(self.device)
            # self.drone_mass = self.drone_mass.flatten(0, -2)[0]
            # gravity_dir, gravity_mag = self.sim.get_physics_context().get_gravity()
            # self.target_z = torch.ones(self.drone.shape, device=self.device) * self.drone_mass * gravity_mag
            self.target_z = torch.zeros(self.drone.shape, device=self.device)
            self.target_z[:] = cfg.goal[3]

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

            return SceneCfg(num_envs=cfg.num_envs, env_spacing=2.5)

        def _reset_idx(self, env_ids: torch.Tensor):
            # since we have multiple parallel environments
            # the environment offset is added to the initial state
            init_state = self.default_init_state[env_ids]
            init_state[:, :3] += self.scene.env_origins[env_ids]

            self.drone.write_root_state_to_sim(init_state, env_ids)

    env = MyEnv(cfg)

    def policy(tensordict: TensorDict):
        target_pitch = env.target_pitch.unsqueeze(1)
        target_roll = env.target_roll.unsqueeze(1)
        target_yaw = env.target_yaw.unsqueeze(1)
        target_z = env.target_z.unsqueeze(1)
        action = torch.cat([target_pitch, target_roll, target_yaw, target_z], dim=-1)
        tensordict["agents", "action"] = action
        # print(f"{tensordict['agents', 'action']=}")
        return tensordict

    def rollout(env: IsaacEnv):
        data_ = env.reset()
        result = []
        for _ in trange(env.max_episode_length):
            data_ = policy(data_)
            data, data_ = env.step_and_maybe_reset(data_)
            result.append(data)
        return torch.stack(result)

    while app.is_running():
        rollout(env)
    app.close()


if __name__ == "__main__":
    main()
