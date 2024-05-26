import random
from collections import deque, namedtuple

import numpy as np
import torch

# Prioritized Experience Replay (PER)
PER_ALPHA = 0.6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, alpha=PER_ALPHA):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            alpha (float): alpha parameter for prioritized replay buffer
        """
        self.buffer_size = buffer_size
        self.action_size = action_size
        self.memory = deque(maxlen=self.buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

        # Prioritized Experience Replay
        self.priorities = np.zeros(self.buffer_size)
        self.max_priority = 1.0
        self.next_idx = 0
        self.current_size = 0
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

        if self.current_size < self.buffer_size:
            self.current_size += 1

        # Add priority
        self.priorities[self.next_idx] = self.max_priority
        self.next_idx = (self.next_idx + 1) % self.buffer_size

    def update_priority(self, idx, priorities):
        self.priorities[idx] = priorities
        self.max_priority = max(np.max(priorities), self.max_priority)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        priorities = (
            self.priorities[: self.current_size] ** self.alpha
        )  # To make the sampling more uniform and reduce overfitting
        sampling_weights = priorities / np.sum(priorities)
        idx_experiences = np.random.choice(
            range(self.current_size),
            size=self.batch_size,
            replace=False,
            p=sampling_weights,
        )  # Sample based on the priority

        idx_adjusted = self.next_idx - self.current_size + idx_experiences
        experiences = [self.memory[idx] for idx in idx_adjusted]

        sampling_weights = (
            torch.from_numpy(sampling_weights[idx_experiences]).float().to(device)
        )
        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

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
        return len(self.memory)
