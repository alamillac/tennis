import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from noise import NormalNoise

from .model import Actor, Critic

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-2  # for soft update of target parameters
LR_ACTOR = 5e-4  # learning rate of the actor
LR_CRITIC = 5e-4  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay
UPDATE_EVERY = 4  # how often to update the network
MIN_BUFFER_SIZE = 1e4  # minimum buffer size before learning

ACTOR_HIDDEN_LAYER_1 = 256
ACTOR_HIDDEN_LAYER_2 = 128

CRITIC_HIDDEN_LAYER_1 = 256
CRITIC_HIDDEN_LAYER_2 = 128

# Gradient clipping
POLICY_MAX_GRAD_NORM = 1.0
VALUE_MAX_GRAD_NORM = 1.0
#VALUE_MAX_GRAD_NORM = float("inf")

# Prioritized Experience Replay (PER)
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_INCREMENT = 0.001
PER_EPSILON = 1e-5

# Noise
POLICY_NOISE_RATIO = 0.1
POLICY_NOISE_CLIP_RATIO = 0.5

# Normal Noise
DECAY_STEPS = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TD3Agent():
    """twin-delayed DDPG (TD3)"""

    def __init__(self, state_size, action_bounds, batch_size=BATCH_SIZE, add_noise=True, n_envs=1, PER=True, noise_decay_steps=DECAY_STEPS):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_bounds (tuple): min and max values of the action
        """
        self.env_min, self.env_max = action_bounds
        action_size = len(self.env_min)
        self.state_size = state_size
        self.action_size = action_size
        self.n_envs = n_envs

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(
            state_size,
            action_bounds,
            fc1_units=ACTOR_HIDDEN_LAYER_1,
            fc2_units=ACTOR_HIDDEN_LAYER_2,
        ).to(device)
        self.actor_target = Actor(
            state_size,
            action_bounds,
            fc1_units=ACTOR_HIDDEN_LAYER_1,
            fc2_units=ACTOR_HIDDEN_LAYER_2,
        ).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(
            state_size,
            action_size,
            fc1_units=CRITIC_HIDDEN_LAYER_1,
            fc2_units=CRITIC_HIDDEN_LAYER_2,
        ).to(device)
        self.critic_target = Critic(
            state_size,
            action_size,
            fc1_units=CRITIC_HIDDEN_LAYER_1,
            fc2_units=CRITIC_HIDDEN_LAYER_2,
        ).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY
        )

        # Noise process
        self.add_noise = add_noise
        self.noise = NormalNoise(action_bounds, decay_steps=noise_decay_steps)

        # Noise ratio
        self.policy_noise_ratio = POLICY_NOISE_RATIO
        self.policy_noise_clip_ratio = POLICY_NOISE_CLIP_RATIO

        # Gradient clipping
        self.policy_max_grad_norm = POLICY_MAX_GRAD_NORM
        self.value_max_grad_norm = VALUE_MAX_GRAD_NORM

        # Replay memory
        self.per = PER
        self.beta = PER_BETA_START
        if self.per:
            self.memory = PrioritizedReplayBuffer(state_size, action_size, BUFFER_SIZE, batch_size, PER_ALPHA)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, batch_size)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Initialize metrics list
        self.metrics = []
        self.actor_learning_steps = 0
        self.critic_learning_steps = 0

        # Num steps
        self.num_steps = 0


    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save noise metrics
        self.num_steps += 1
        self.metrics.append(("noise_ratio", self.noise.noise_ratio, self.num_steps))
        self.metrics.append(("ratio_noise_injected", self.noise.ratio_noise_injected, self.num_steps))

        # Save experience / reward
        if self.n_envs == 1:
            self.memory.add(states, actions, rewards, next_states, dones)
        else:
            for state, action, reward, next_state, done in zip(
                states, actions, rewards, next_states, dones
            ):
                self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) < MIN_BUFFER_SIZE:
            # Skip learning if not enough samples are available in memory
            return

        # Get random subset and learn
        experiences = self.memory.sample()

        # Critic learn every time step if enough samples are available in memory
        # ---------------------------- update critic ---------------------------- #
        critic_loss = self._critic_learn(experiences, GAMMA)
        self.critic_learning_steps += 1
        self.metrics.append(("critic_loss", critic_loss, self.critic_learning_steps))

        # Actor learn every UPDATE_EVERY time steps and if enough samples are available in memory
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # ---------------------------- update actor ---------------------------- #
            actor_loss = self._actor_learn(experiences)
            self.actor_learning_steps += 1
            self.metrics.append(("actor_loss", actor_loss, self.actor_learning_steps))

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

            # ------------------- update beta ------------------- #
            if self.per:
                self.beta = min(1.0, self.beta + PER_BETA_INCREMENT)


    def act_train(self, states):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if self.add_noise:
            actions += self.noise.sample(self.n_envs)
        return np.clip(actions, self.env_min, self.env_max)

    def act(self, states):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if self.add_noise:
            noise = self.noise.sample(self.n_envs, noise_ratio=0.05)
            #noise = self.noise.sample().reshape(actions.shape)
            actions += noise
        return np.clip(actions, self.env_min, self.env_max)

    def reset(self):
        self.noise.step()

    def pop_metrics(self):
        metrics = self.metrics
        self.metrics = []
        return metrics

    def _actor_learn(self, experiences):
        if self.per:
            states, _, _, _, _, _, _ = experiences
        else:
            states, _, _, _, _= experiences

        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local.Qa(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor_local.parameters(), self.policy_max_grad_norm
        )  # Gradient clipping
        self.actor_optimizer.step()

        return actor_loss.item()


    def _critic_learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        if self.per:
            states, actions, rewards, next_states, dones, sampling_weights, exp_idx = (
                experiences
            )
        else:
            states, actions, rewards, next_states, dones = (
                experiences
            )

        with torch.no_grad():
            env_min = self.actor_target.env_min
            env_max = self.actor_target.env_max
            a_ran = env_max - env_min
            a_noise = torch.randn_like(actions) * self.policy_noise_ratio * a_ran
            n_min = env_min * self.policy_noise_clip_ratio
            n_max = env_max * self.policy_noise_clip_ratio
            a_noise = torch.max(torch.min(a_noise, n_max), n_min)

            # Get predicted next-state actions and Q values from target models
            actions_next = self.actor_target(next_states)
            noisy_actions_next = actions_next + a_noise
            noisy_actions_next = torch.max(torch.min(noisy_actions_next, env_max), env_min)

            Q_targets_next_a, Q_targets_next_b = self.critic_target(next_states, actions_next)
            Q_targets_next = torch.min(Q_targets_next_a, Q_targets_next_b)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected_a, Q_expected_b = self.critic_local(states, actions)

        # As we are using Prioritized Experience Replay (PER), we need to multiply the loss by the importance sampling weights
        if self.per:
            buffer_size = len(self.memory)
            weights = (buffer_size * sampling_weights) ** (
                -self.beta
            )  # Importance sampling weights
            weights = weights / weights.max()  # Normalize the weights

            critic_weighted_loss_a = (
                weights * F.mse_loss(Q_expected_a, Q_targets, reduction="none")
            ).mean()

            critic_weighted_loss_b = (
                weights * F.mse_loss(Q_expected_b, Q_targets, reduction="none")
            ).mean()
            critic_loss = critic_weighted_loss_a + critic_weighted_loss_b
        else:
            critic_loss_a = F.mse_loss(Q_expected_a, Q_targets)
            critic_loss_b = F.mse_loss(Q_expected_b, Q_targets)
            critic_loss = critic_loss_a + critic_loss_b

        if self.per:
            # Update the priorities
            td_errors_a = torch.abs(Q_expected_a - Q_targets).detach().squeeze().cpu().numpy()
            td_errors_b = torch.abs(Q_expected_b - Q_targets).detach().squeeze().cpu().numpy()
            #td_errors = np.maximum(td_errors_a, td_errors_b)
            td_errors = td_errors_a + td_errors_b
            self.memory.update_priority(exp_idx, td_errors + PER_EPSILON)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic_local.parameters(), self.value_max_grad_norm
        )  # Gradient clipping
        self.critic_optimizer.step()

        return critic_loss.item()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def save(self, path):
        torch.save(self.actor_local.state_dict(), path)

    def load(self, path):
        self.actor_local.load_state_dict(torch.load(path))
        self.actor_target.load_state_dict(torch.load(path))

    def get_state(self):
        return {
            "actor_local_state_dict": self.actor_local.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_local_state_dict": self.critic_local.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "noise_ratio": self.noise.noise_ratio,
            "beta": self.beta,
        }

    def load_state(self, state):
        self.actor_local.load_state_dict(state["actor_local_state_dict"])
        self.actor_target.load_state_dict(state["actor_target_state_dict"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])

        self.critic_local.load_state_dict(state["critic_local_state_dict"])
        self.critic_target.load_state_dict(state["critic_target_state_dict"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
        self.noise.noise_ratio = state["noise_ratio"]
        self.beta = state["beta"]
