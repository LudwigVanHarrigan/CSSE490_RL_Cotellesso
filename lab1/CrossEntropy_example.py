# %% [markdown]
# # Lab 1: Cross-Entropy Method (CEM) for CartPole
# 
# ## Introduction
# In this lab, we will implement the Cross-Entropy Method (CEM) algorithm to solve the CartPole-v1 environment. CEM is a simple yet effective policy search method that iteratively improves a policy by sampling from a distribution and selecting elite samples.
# 
# ## Student Information
# **Name:** Kevin Cotellesso  
# **Date:** 3-10-2026
#
# ---

# %% [markdown]
# ## Part 1: Setup and Environment Testing
# Let's start by importing the necessary libraries and creating the CartPole environment.

# %%
import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display

# Create environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

total_reward = 0.0
total_steps = 0
obs, info = env.reset()

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Initial observation: {obs}")

# %% [markdown]
# ### Random Agent Test
# Let's test the environment with a random agent to see how it performs.

# %%
while True:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    plt.imshow(env.render())
    display.display(plt.gcf())    
    display.clear_output(wait=True)
    plt.close()

    if done or truncated:
        break

print(f"Random agent - Total reward: {total_reward}, Total steps: {total_steps}")


# %% [markdown]
# ---
# ## Part 2: Linear Policy Implementation
# 
# We'll implement a simple linear policy that maps observations to actions using a weight matrix W and bias vector b.

# %%
class LinearPolicy(object):
    def __init__(self, theta, ob_space, ac_space):
        assert len(theta) == (ob_space + 1) * ac_space
        self.W = theta[0:ob_space*ac_space].reshape(ob_space, ac_space)
        self.b = theta[ob_space*ac_space:None].reshape(1, ac_space)

    def act(self, obs):
        y = obs.dot(self.W) + self.b
        a = y.argmax()
        return a


# Test the policy
ob_space = env.observation_space.shape[0]
ac_space = env.action_space.n
n_theta = (ob_space + 1) * ac_space
print(f"Policy parameter size: {n_theta}")

def run_episode(policy, env, num_steps, render=False):
    total_rew = 0
    ob, info = env.reset()
    for t in range(num_steps):
        a = policy.act(ob)
        ob, reward, done, truncated, info = env.step(a)
        total_rew += reward
        if done or truncated:
            break
    return total_rew


theta_mean = np.zeros(n_theta)
theta_std = np.ones(n_theta)

reward_list = []

# CEM hyperparameters
n_iterations = 50  # Number of CEM iterations
n_samples = 25     # Number of policy samples per iteration
n_elite = 5        # Number of elite samples to keep

print("Starting CEM training...")
print(f"Iterations: {n_iterations}, Samples: {n_samples}, Elite: {n_elite}")
print("-" * 60)

for itr in range(n_iterations):
    # Sample policies from current distribution
    thetas = np.random.multivariate_normal(
        mean=theta_mean,
        cov=np.diag(np.array(theta_std)**2),
        size=n_samples
    )

    rewards = []
    for theta in thetas:
        policy = LinearPolicy(theta, ob_space, ac_space)
        r = run_episode(policy, env, 500)
        rewards.append(r)

    rewards = np.array(rewards)
    
    # Get elite parameters
    elite_inds = rewards.argsort()[-n_elite:]
    elite_thetas = thetas[elite_inds]

    # Update theta_mean, theta_std
    theta_mean = elite_thetas.mean(axis=0)
    theta_std = elite_thetas.std(axis=0)
    
    # Log progress
    print(f"[Iteration {itr:2d}] mean: {np.mean(rewards):5.1f} | max: {np.max(rewards):5.1f} | min: {np.min(rewards):5.1f}")
    reward_list.append(np.mean(rewards))

print("-" * 60)
print("Training complete!")

env.close()

# %% [markdown]
# ---
# ## Part 5: Results Analysis
# 
# Let's visualize the learning curve and analyze the results.

# %%
# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(reward_list, linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Average Reward', fontsize=12)
plt.title('CEM Learning Curve on CartPole-v1', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print statistics
print(f"Initial average reward: {reward_list[0]:.2f}")
print(f"Final average reward: {reward_list[-1]:.2f}")
print(f"Best average reward: {max(reward_list):.2f}")
print(f"Improvement: {reward_list[-1] - reward_list[0]:.2f}")




# %% [markdown]
# ---
# ## Save Results
# 
# Save the results to a CSV file for later analysis.

# %%
df = pd.DataFrame({"reward": reward_list})
df.to_csv("Linear_CEM.csv", index=False, header=True)

env.close()
# %%
