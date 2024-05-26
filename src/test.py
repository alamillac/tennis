import numpy as np

from environment import Env
from td3 import TD3Agent
from trainer import Trainer

env_path = {
    "tennis": (
        "Tennis_Linux/Tennis.x86_64",
        "tennis.pth",
    ),
    "soccer": (
        "Soccer_Linux/Soccer.x86_64",
        "soccer.pth",
    ),
}


env_name = "tennis"
#env_name = "soccer"

env_filename, save_path = env_path[env_name]

env = Env(env_filename, train_mode=False, seed=0)
low = np.array([-1] * env.action_size)
high = np.array([1] * env.action_size)
action_bounds = (low, high)

trainer = Trainer(max_t=1000)

agent = TD3Agent(
    state_size=env.state_size, action_bounds=action_bounds, n_envs=env.num_agents
)
agent.load(save_path)

trainer.test(env, agent, num_episodes=2)
