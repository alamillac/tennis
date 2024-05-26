from datetime import datetime
from os import path

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

from environment import Env
from td3 import TD3Agent
from trainer import Trainer

env_path = {
    "tennis": (
        "Tennis_Linux/Tennis.x86_64",
        "tennis.pth",
        "tennis_checkpoint.pth",
        8000,
        128,
    ),
    "soccer": (
        "Soccer_Linux/Soccer.x86_64",
        "soccer.pth",
        "soccer_checkpoint.pth",
        1000,
        128,
    )
}

env_name = "tennis"
#env_name = "soccer"
env_filename, save_path, save_checkpoint_path, max_episodes, batch_size = env_path[
    env_name
]

env = Env(env_filename, train_mode=True)
low = np.array([-1] * env.action_size)
high = np.array([1] * env.action_size)
action_bounds = (low, high)

print("Number of agents:", env.num_agents)
print("Number of actions:", env.action_size)
print("States have length:", env.state_size)

log_dir = path.join(
    "runs",
    "experiment_{}_{}".format(env_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
)
writer = SummaryWriter(log_dir)

trainer = Trainer(
    max_episodes=max_episodes,
    max_t=1000,
    save_model_path=save_path,
    save_checkpoint_path=save_checkpoint_path,
    override_checkpoint=False,
    writer=writer,
)

agent = TD3Agent(
    state_size=env.state_size,
    action_bounds=action_bounds,
    batch_size=batch_size,
    n_envs=env.num_agents,
    noise_decay_steps=2000,
)

scores = trainer.train_until(env, agent, desired_score=0.5, consecutive_episodes=100)
trainer.plot_scores(scores)
