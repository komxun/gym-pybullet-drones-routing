"""Replay buffer with proper numpy-based storage."""

import numpy as np


class ReplayBuffer:
    """Fixed-size circular replay buffer for off-policy DRL.

    Stores transitions as (state, action, reward, next_state, is_terminal).
    Uses pre-allocated numpy arrays for efficiency.
    """

    def __init__(self, max_size: int = 200000, batch_size: int = 64):
        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0

        # Pre-allocate with object arrays; shapes are set on first store
        self._ss = np.empty(max_size, dtype=np.ndarray)
        self._as = np.empty(max_size, dtype=np.ndarray)
        self._rs = np.empty(max_size, dtype=np.ndarray)
        self._ns = np.empty(max_size, dtype=np.ndarray)
        self._ds = np.empty(max_size, dtype=np.ndarray)

    def store(self, transition: tuple):
        """Store a single (s, a, r, s', done) transition."""
        s, a, r, ns, d = transition
        self._ss[self._idx] = s
        self._as[self._idx] = a
        self._rs[self._idx] = r
        self._ns[self._idx] = ns
        self._ds[self._idx] = d

        self._idx = (self._idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int = None) -> tuple:
        """Sample a random batch.

        Returns
        -------
        tuple of np.ndarray
            (states, actions, rewards, next_states, is_terminals)
            Each is a 2D array of shape (batch_size, feature_dim).

        NOTE: The order here MUST match what the network's load_experiences()
        expects. This was a critical bug in the old code.
        """
        if batch_size is None:
            batch_size = self.batch_size

        idxs = np.random.choice(self.size, batch_size, replace=False)

        states = np.vstack(self._ss[idxs])
        actions = np.vstack(self._as[idxs])
        rewards = np.vstack(self._rs[idxs])
        next_states = np.vstack(self._ns[idxs])
        is_terminals = np.vstack(self._ds[idxs])

        return states, actions, rewards, next_states, is_terminals

    def __len__(self):
        return self.size
