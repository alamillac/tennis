import random

from unityagents import UnityEnvironment

MAX_SEED_RANGE = 100000


class Env:
    def __init__(self, file_name, train_mode=True, seed=None):
        self.train_mode = train_mode
        no_graphics = self.train_mode

        if seed is None:
            seed = random.randrange(MAX_SEED_RANGE)

        self.env = UnityEnvironment(
            file_name=file_name, no_graphics=no_graphics, seed=seed
        )

        # get the default brain
        self.brain_name = self.env.brain_names[0]

        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]

        # Number of agents in the environment
        self.num_agents = len(env_info.agents)

        # states space
        states = self._get_states(env_info)
        self.state_size = states.shape[1]

        # number of actions
        brain = self.env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size

    def _get_env_info(self, env_info):
        next_states = self._get_states(env_info)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done
        return next_states, rewards, dones

    def _get_states(self, env_info):
        return env_info.vector_observations  # For each agent

    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        return self._get_states(env_info)

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        return self._get_env_info(env_info)

    def close(self):
        self.env.close()
