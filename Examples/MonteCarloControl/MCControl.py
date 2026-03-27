# %% [markdown]
# # Monte Carlo Control on FrozenLake
#
# In dynamic programming (policy/value iteration), we had access to the full
# transition model `env.unwrapped.P`. In practice, we rarely know the environment
# dynamics. Monte Carlo Control learns a policy by **sampling episodes** and
# using the observed returns to estimate Q-values — no model needed.
#
# The key idea: run episodes with the current policy, compute returns from the
# collected rewards, and update Q(s,a) toward the observed returns.

# %%
import numpy as np
from tqdm import tqdm
import gymnasium as gym

env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
n_state = env.unwrapped.observation_space.n
n_action = env.unwrapped.action_space.n
print("# of actions:", n_action)
print("# of states:", n_state)

# %% [markdown]
# ## Step 1: Collect an Episode
#
# Unlike DP methods that sweep over all states using the model, MC methods
# must **run a full episode** to collect (state, action, reward) trajectories.
# We then compute returns from these trajectories.

# %%


def run_episode(env, policy):
    """Run one episode and return the trajectory (states, actions, rewards)."""
    obs = env.reset()[0]
    states, actions, rewards = [], [], []
    while True:
        states.append(obs)
        action = int(policy(obs))
        actions.append(action)
        obs, reward, done, truncated, _ = env.step(action)
        rewards.append(reward)
        if done or truncated:
            break
    return states, actions, rewards


# %% [markdown]
# ## Step 2: Policy with Epsilon-Greedy Exploration
#
# MC Control uses **epsilon-greedy** exploration: with probability epsilon, pick
# a random action; otherwise, pick the greedy action from Q.
#
# Why do we need Q instead of V here? In DP, we could extract a policy from V
# because we had access to the transition model P(s'|s,a). Without a model,
# we can't compute which action leads to higher-value states — so we must
# learn Q(s,a) directly and pick the action with the highest Q.

# %%


class EpsilonGreedyPolicy:
    def __init__(self, Q, N, eps, gamma=0.98):
        self.Q = Q          # Q-table: Q[state, action]
        self.N = N          # Visit count: N[state, action]
        self.eps = eps       # Exploration rate
        self.gamma = gamma   # Discount factor

    def update(self, states, actions, rewards):
        """Update Q-values using returns computed from the episode."""
        # Compute returns G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
        # We compute this backwards for efficiency
        returns = np.zeros(len(rewards))
        G = 0
        for i in reversed(range(len(rewards))):
            G = rewards[i] + self.gamma * G
            returns[i] = G

        # Update Q using incremental mean: Q = Q + (G - Q) / N
        for state, action, G in zip(states, actions, returns):
            self.N[state, action] += 1
            self.Q[state, action] += (G - self.Q[state, action]) / self.N[state, action]

    def __call__(self, state):
        """Select action using epsilon-greedy."""
        if np.random.rand() < self.eps:
            return np.random.choice(n_action)
        return np.argmax(self.Q[state, :])


# %% [markdown]
# ## Step 3: Training Loop
#
# We repeatedly: (1) collect an episode, (2) update Q from returns, and
# (3) decay epsilon so the policy gradually shifts from exploration to exploitation.

# %%
n_episodes = 80000
Q = np.zeros((n_state, n_action))
N = np.zeros_like(Q)
policy = EpsilonGreedyPolicy(Q, N, eps=1.0)

for i in tqdm(range(n_episodes)):
    states, actions, rewards = run_episode(env, policy)
    policy.update(states, actions, rewards)
    # Linear decay: epsilon goes from 1.0 to 0.01 over training
    policy.eps = max(0.01, policy.eps - 1.0 / n_episodes)

# %% [markdown]
# ## Evaluate the Learned Policy
#
# Set epsilon to 0 (fully greedy) and run 100 episodes to measure performance.

# %%
policy.eps = 0.0
scores = [sum(run_episode(env, policy)[2]) for _ in range(100)]
print("Final score: {:.2f}".format(np.mean(scores)))
