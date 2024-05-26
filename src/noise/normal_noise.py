import numpy as np


# Only makes sense to use this noise for continuous action spaces
class NormalNoise:
    """Normal noise process."""

    def __init__(
        self, bounds, init_noise_ratio=0.5, min_noise_ratio=0.05, decay_steps=1000
    ):
        """Initialize parameters and noise process."""
        self.low, self.high = bounds
        self.size = len(self.low)
        self.noise_ratio = init_noise_ratio
        self.min_noise_ratio = min_noise_ratio
        self.increment = (min_noise_ratio - init_noise_ratio) / decay_steps
        self.ratio_noise_injected = 0

    def step(self):
        """Update the noise ratio."""
        self.noise_ratio = max(self.min_noise_ratio, self.noise_ratio + self.increment)

    def sample(self, num_samples=1, noise_ratio=None):
        """Return a noise sample."""
        noise_ratio = noise_ratio or self.noise_ratio
        scale = (self.high - self.low) * noise_ratio
        scale = scale[:, np.newaxis]
        sample = np.random.normal(loc=0, scale=scale, size=(self.size, num_samples)).T
        self.ratio_noise_injected = np.mean(np.abs(sample / (self.high - self.low)))
        return sample
