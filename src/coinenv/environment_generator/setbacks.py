import random

from minigrid.minigrid_env import MiniGridEnv

from coinenv.environment_generator.utils import get_all_dead_ends


class SetbackFactory:
    def __init__(self):
        self.setbacks = {
            "return_to_start": self.return_to_start,
            "rotate_agent": self.rotate_agent,
            "teleport_to_dead_end": self.teleport_to_dead_end,
        }

    def return_to_start(self, env: MiniGridEnv) -> MiniGridEnv:
        env.agent_pos = env.agent_start_pos
        env.agent_dir = env.agent_start_dir
        return env

    def rotate_agent(self, env: MiniGridEnv) -> MiniGridEnv:
        env.agent_dir = (env.agent_dir + random.randint(1, 3)) % 4
        return env

    def teleport_to_dead_end(self, env: MiniGridEnv) -> MiniGridEnv:
        env.agent_pos = random.choice(get_all_dead_ends(env))
        return env

    def get_all_setbacks(self, env: MiniGridEnv) -> list[MiniGridEnv]:
        return [self.setbacks[setback](env) for setback in self.setbacks]
