import time

import numpy as np

from unityagents import UnityEnvironment

# please do not modify the line below
env = UnityEnvironment(file_name="../VisualBanana_Linux/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print("Number of agents:", len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print("Number of actions:", action_size)

# examine the state space
state = env_info.visual_observations[0]
print("States look like:")
state_size = state.shape
print("States have shape:", state.shape)

env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
for i in range(1, 100):
    state = env_info.visual_observations[0]  # get the current state
    score = 0  # initialize the score
    step = 0
    while True:
        action = np.random.randint(action_size)  # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.visual_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        step += 1
        #time.sleep(0.1)
        print(f"\rEpisode {i} Step {step} Score: {score}", end="")
        if done:  # exit loop if episode finished
            print(f"\rEpisode {i} Step {step} Score: {score}")
            break
    if i % 50 == 0:
        env_info = env.restart(train_mode=True)[brain_name]
    else:
        env_info = env.reset(train_mode=True)[brain_name]

env.close()
