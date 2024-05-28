import random
from collections import deque, namedtuple

import numpy as np
import torch

# Prioritized Experience Replay (PER)
PER_ALPHA = 0.6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
        self, state_size, action_size, buffer_size, batch_size, alpha=PER_ALPHA
    ):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            alpha (float): alpha parameter for prioritized replay buffer
        """
        self.buffer_size = buffer_size
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size

        # Memory
        self.states = np.zeros((self.buffer_size, self.state_size))
        self.actions = np.zeros((self.buffer_size, self.action_size))
        self.rewards = np.zeros((self.buffer_size, 1))
        self.next_states = np.zeros((self.buffer_size, self.state_size))
        self.dones = np.zeros((self.buffer_size, 1)).astype(np.uint8)

        # Prioritized Experience Replay
        self.priority_weights = np.zeros(self.buffer_size)
        self.max_priority = 1.0
        self.next_idx = 0
        self.current_size = 0
        self.alpha = alpha
        self.sum_weights = 0.0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # Save into memory
        self.states[self.next_idx] = state
        self.actions[self.next_idx] = action
        self.rewards[self.next_idx] = reward
        self.next_states[self.next_idx] = next_state
        self.dones[self.next_idx] = done

        if self.current_size < self.buffer_size:
            self.current_size += 1

        # Add priority
        new_priority = self.max_priority
        new_weight = (
            new_priority**self.alpha
        )  # To make the sampling more uniform and reduce overfitting
        self.sum_weights += new_weight
        self.priority_weights[self.next_idx] = new_weight

        # Update the next index
        self.next_idx = (self.next_idx + 1) % self.buffer_size

    def update_priority(self, idx, new_priorities):
        new_weights = new_priorities**self.alpha
        self.sum_weights += np.sum(new_weights)  # We add back the new weights
        self.priority_weights[idx] = new_weights
        self.max_priority = max(np.max(new_priorities), self.max_priority)

    def get_experiences(self, idxs):
        states = self.states[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_states = self.next_states[idxs]
        dones = self.dones[idxs]
        return states, actions, rewards, next_states, dones

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        weights = self.priority_weights[
            : self.current_size
        ]  # Get the weights of the experiences

        prob_weights = weights / self.sum_weights  # Normalize the weights
        try:
            idx_experiences = np.random.choice(
                range(self.current_size),
                size=self.batch_size,
                replace=False,
                p=prob_weights,
            )  # Sample based on the priority
        except:
            self.sum_weights = np.sum(
                weights
            )  # Error in sampling. Reset the sum of weights
            prob_weights = weights / self.sum_weights  # Normalize the weights
            idx_experiences = np.random.choice(
                range(self.current_size),
                size=self.batch_size,
                replace=False,
                p=prob_weights,
            )  # Sample based on the priority

        self.sum_weights -= np.sum(
            weights[idx_experiences]
        )  # Remove the sampled weights from the sum of weights. We will add them back with the new priorities

        states, actions, rewards, next_states, dones = self.get_experiences(
            idx_experiences
        )

        sampling_weights = (
            torch.from_numpy(prob_weights[idx_experiences]).float().to(device)
        )  # Get the weights of the sampled experiences

        states = torch.from_numpy(states).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).int().to(device)
        actions = torch.from_numpy(actions).float().to(device)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            sampling_weights,
            idx_experiences,
        )

    def __len__(self):
        """Return the current size of internal memory."""
        return self.current_size
