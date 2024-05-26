import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

LINE_CLEAR = "\x1b[2K"


class Trainer:
    def __init__(
        self,
        max_episodes=2000,
        max_t=1000,
        save_every=100,
        save_checkpoint_path="checkpoint.pth",
        save_model_path="model.pth",
        override_checkpoint=False,
        disable_bar_progress=False,
        writer=None,
    ):
        """Deep Q-Learning.

        Params
        ======
            max_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
        """
        self.max_t = max_t
        self.max_episodes = max_episodes
        self.save_checkpoint_path = save_checkpoint_path
        self.save_model_path = save_model_path
        self.save_every = save_every
        self.override_checkpoint = override_checkpoint
        self.writer = writer
        self.disable_bar_progress = disable_bar_progress

    def plot_scores(self, scores):
        # plot the scores
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel("Score")
        plt.xlabel("Episode #")
        plt.show()

    def test(self, env, agent, num_episodes=100):
        scores = []
        max_score = -np.Inf
        min_score = np.Inf
        total_score = 0
        episode_bar = trange(num_episodes, disable=self.disable_bar_progress)
        for i_episode in episode_bar:
            states = env.reset()
            score = 0
            steps_bar = trange(
                self.max_t, leave=False, disable=self.disable_bar_progress
            )
            for step in steps_bar:
                actions = agent.act(states)
                states, rewards, dones = env.step(actions)
                score += np.mean(rewards)
                steps_bar.set_description(f"Step {step + 1} Score: {score:.2f}")
                if self.disable_bar_progress:
                    print(f"\rStep {step + 1} Score: {score:.2f}", end="")
                done = np.any(dones)  # if any agent is done, then the episode is done
                if done:
                    break

            scores.append(score)
            max_score = max(score, max_score)
            min_score = min(score, min_score)
            total_score += score
            avg_score = total_score / (i_episode + 1)
            episode_bar.set_description(
                f"Episode {i_episode + 1} Scores: Last {score:.2f} Min {min_score:.2f} Max {max_score:.2f} Avg {avg_score:.2f}"
            )
            if self.disable_bar_progress:
                print(
                    f"\rEpisode {i_episode + 1} Scores: Last {score:.2f} Min {min_score:.2f} Max {max_score:.2f} Avg {avg_score:.2f}"
                )
        env.close()
        return scores

    def _train(self, env, agent):
        scores_window_5 = deque(maxlen=5)  # last scores
        scores_window_100 = deque(maxlen=100)  # last scores
        init_episode = self.load_checkpoint(self.save_checkpoint_path, agent)
        episode_bar = trange(
            init_episode, self.max_episodes, disable=self.disable_bar_progress
        )
        for i_episode in episode_bar:
            states = env.reset()
            agent.reset()
            score = 0
            steps_bar = trange(
                self.max_t, leave=False, disable=self.disable_bar_progress
            )
            for step in steps_bar:
                actions = agent.act_train(states)
                next_states, rewards, dones = env.step(actions)
                agent.step(states, actions, rewards, next_states, dones)
                states = next_states
                score += np.mean(rewards)
                steps_bar.set_description(f"Step {step + 1} Score: {score:.2f}")
                done = np.any(dones)  # if any agent is done, then the episode is done
                if done:
                    break

            scores_window_5.append(score)  # save most recent score
            scores_window_100.append(score)  # save most recent score
            avg_score_5 = np.mean(scores_window_5)
            avg_score_100 = np.mean(scores_window_100)
            episode_bar.set_description(
                f"Episode {i_episode + 1} Score [{score:.2f} {avg_score_5:.2f} {avg_score_100:.2f}]"
            )
            if self.disable_bar_progress:
                score_str = f"\rEpisode {i_episode + 1} Score [{score:.2f} {avg_score_5:.2f} {avg_score_100:.2f}]"
                print(score_str, end="")
                if i_episode % 10 == 0:
                    print(score_str)

            if self.writer:
                self.writer.add_scalar("score", score, i_episode)
                agent_metrics = agent.pop_metrics()
                for metric_label, metric, idx in agent_metrics:
                    self.writer.add_scalar(metric_label, metric, idx)

            # Save the checkpoint
            if (i_episode + 1) % self.save_every == 0:
                self.save_checkpoint(self.save_checkpoint_path, agent, i_episode + 1)

            yield i_episode, score

    def save_checkpoint(self, path, agent, i_episode):
        agent_state = agent.get_state()
        checkpoint = {
            "i_episode": i_episode,
            "agent_state": agent_state,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, agent):
        # Check if the file exists
        if not os.path.isfile(path) or self.override_checkpoint:
            return 0

        checkpoint = torch.load(path)
        agent.load_state(checkpoint["agent_state"])

        return checkpoint["i_episode"]

    def train_until(self, env, agent, desired_score, consecutive_episodes=100):
        scores_window = deque(maxlen=consecutive_episodes)  # last scores
        scores = []  # list containing scores from each episode
        for i_episode, score in self._train(env, agent):
            scores.append(score)

            scores_window.append(score)  # save most recent score
            avg_score = np.mean(scores_window)

            if avg_score >= desired_score and i_episode > consecutive_episodes:
                print(
                    f"\nEnvironment solved in {i_episode} episodes!\tAverage Score: {avg_score:.2f}"
                )
                break

        agent.save(self.save_model_path)
        env.close()
        return scores

    def train(self, env, agent):
        scores = []  # list containing scores from each episode
        for _, score in self._train(env, agent):
            scores.append(score)

        agent.save(self.save_model_path)
        env.close()
        return scores
