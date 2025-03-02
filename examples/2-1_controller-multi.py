import hydra
import os
import torch
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from tensordict import TensorDict


@hydra.main(config_path=os.path.dirname(__file__), config_name="demo_2")
def main(cfg):
    app = init_simulation_app(cfg)

    # due to the design of Isaac Sim, these imports are only available
    # after the SimulationApp instance is created
    from omni_drones.envs.isaac_env import IsaacEnv
    from omni_drones.robots.assets import Multirotor, HUMMINGBIRD_CFG

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
            # now it accounts for multiple drones
            self.drone: Multirotor = self.scene["drone"]
            self.default_init_state = self.drone.data.default_root_state.clone()
            self.default_init_state[:, :, 1] += torch.arange(self.drone.shape[1], device=self.device)

            # set target positions for the drones
            self.target_pos = self.default_init_state[:, :, :3].clone()
            self.target_pos[:, :, 0] -= torch.arange(self.drone.shape[1], device=self.device) * 1
            self.target_pos[:, :, 2] += torch.arange(self.drone.shape[1], device=self.device) * 0.5
            self.target_yaw = torch.zeros(self.drone.shape, device=self.device)
            self.target_yaw[:] = torch.pi / 2

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

                # this time, we spawn two drones by providing a list of `prim_paths`
                # note that they should **share the same path prefix**
                drone = HUMMINGBIRD_CFG.replace(
                    prim_path=[
                        "{ENV_REGEX_NS}/Robot_0",
                        "{ENV_REGEX_NS}/Robot_1",
                        "{ENV_REGEX_NS}/TheThridDrone", # this works
                        # "{ENV_REGEX_NS}/Something/Robot_2", # this will not work
                    ],
                )

            return SceneCfg(num_envs=cfg.num_envs, env_spacing=2.5)

        def _reset_idx(self, env_ids: torch.Tensor):
            # since we have multiple parallel environments
            # the environment offset is added to the initial state
            init_state = self.default_init_state[env_ids]
            init_state[:, :, :3] += self.scene.env_origins[env_ids].unsqueeze(1)

            # NOTE: the difference to the single drone case, where we used
            #   init_state = self.default_init_state[env_ids]
            #   init_state[:, :3] += self.scene.env_origins[env_ids]

            env_ids = self.drone.resolve_ids(env_ids)
            self.drone.write_root_state_to_sim(init_state.flatten(0, 1), env_ids)

    env: MyEnv = MyEnv(cfg)

    def policy(tensordict: TensorDict):
        root_pos_e = (
            env.drone.data.root_pos_w.reshape(*env.drone.shape, 3)
            - env.scene.env_origins.unsqueeze(1)
        )
        target_pos = env.target_pos - root_pos_e
        target_yaw = env.target_yaw.unsqueeze(-1)
        action = torch.cat([target_pos, target_yaw], dim=-1)
        tensordict["agents", "action"] = action
        return tensordict

    tensordict = env.reset()

    while True:
        tensordict = policy(tensordict)
        # torchrl automatically handles stepping and reset for us
        _, tensordict = env.step_and_maybe_reset(tensordict)


if __name__ == "__main__":
    main()
