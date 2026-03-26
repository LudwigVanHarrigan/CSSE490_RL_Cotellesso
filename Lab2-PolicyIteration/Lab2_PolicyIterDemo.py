# %% [markdown]
# 

# %% [markdown]
#  In this lab, we will implement policy iteration and value iteration to solve the Frozen lake problem.
# 
# 
# 
#  Let's first get ourselves familarized with the environment.

# %%
import numpy as np
import gymnasium as gym

env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)  # or you can try '8x8'
env.reset()
n_state = env.unwrapped.observation_space.n
n_action = env.unwrapped.action_space.n
print("# of actions", n_action)
print("# of states", n_state)
P = env.unwrapped.P
# At state 14 apply action 2
for prob, next_state, reward, done in P[14][2]:
    print("If apply action 2 under state 14, there is %3.2g probability it will transition to state %d and yield reward as %i" % (
        prob, next_state, reward))


# %% [markdown]
#  First, let's write utility functions we will use to run experiments.

# %% [markdown]
#  Given certain policy, how can we compute the value function for each state.

# %%


def compute_policy_v(env, policy, gamma=1.0):
    # The goal of this function is to calculate the Q(s,a) for each action under each state
    eps = 1e-10
    V = np.zeros(n_state)
    while True:
        prev_V = np.copy(V)
        for state in range(n_state):
            action = policy[state] # Deterministic!
            V[state] = 0
            for prob, next_state, reward, done in P[state][action]:
                V[state] += prob * (reward + gamma * prev_V[next_state])
                # V is sum of all Q(s,a) weighted by probability of taking the action a.
                # In our case, after one iteration, our policy is deterministic so we just have to 
                # include the Q from the next_state.
        if (np.sum((np.fabs(prev_V - V))) <= eps):
            break
    return V



# %% [markdown]
#  Let's test a random policy

# %%
random_policy = [np.random.choice(n_action) for _ in range(n_state)]
print(random_policy)
rand_v = compute_policy_v(env, random_policy)
print(np.round(rand_v, 5))


# %% [markdown]
#  Given value function, we need to extract the best policy from it.

# %%

# need a Q here, since we are getting the best policy by argmaxing
def extract_policy(V, gamma=1.0):
    policy = np.zeros(n_state) # Action for each state
    Q = np.zeros((n_state, n_action))
    for state in range(n_state):
        for action in range(n_action):
            for prob, next_state, reward, done in P[state][action]:
                Q[state, action] += prob * (reward + gamma*V[next_state])
        policy[state] = np.argmax(Q[state]) # Best action for each state
    return policy



# %%
print(extract_policy(rand_v))


# %% [markdown]
#  Now let's start with a random policy and compute the value then extract new policy. Do this recursively will improve the policy

# %%


def run_episode(env, policy, gamma=1.0, render=False):
    """ Runs an episode and return the total reward """
    obs, _ = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, truncated, done, _ = env.step(int(policy[obs]))
        # this will calculate the return for the first step
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done or truncated:
            break
    return total_reward



# %%

# Policy iteration: 
# 1. Given a deterministic policy (state -> action), compute value of every state
# 2. Given these values, get a new policy.
# 3. Run an episode to get a score
# Repeat until policy does not change.
policy_PI = random_policy
max_iterations = 200000
gamma = 1.0
for i in range(max_iterations):
    policy_v = compute_policy_v(env, policy_PI)
    new_policy = extract_policy(policy_v)
    score = run_episode(env, new_policy, gamma, False)
    print("iteration %i: score %f" % (i, score))
    if (np.all(policy_PI == new_policy)):
        print('Policy-Iteration converged at step %d.' % (i+1))
        break
    policy_PI = new_policy

scores = [run_episode(env, policy_PI, gamma, False) for _ in range(1000)]
print("PI Final score:", np.mean(scores))

# %%
# Value iteration: Converge V then find Q then find policy.
policy_VI = random_policy
max_iterations = 2000
gamma = 1.0
eps = 1e-10
Vs = np.zeros((max_iterations, n_state))
for iter in range(1, max_iterations): 
# 1. Loop through each state.
    Q = np.zeros((n_state, n_action)) # Reset this between iterations
    for state in range(n_state):
        # Get the Q-function for each action
        for action in range(n_action):
            for prob, next_state, reward, done in P[state][action]:
                Q[state, action] += prob * (reward + gamma*Vs[iter-1][next_state])
        Vs[iter][state] = np.max(Q[state]) # The next iteration of this state is the max value of Q (all possible actions)
    # Check to see if we've converged
    if(np.sum(np.abs(Vs[iter]-Vs[iter-1])) < eps):
        print(f"Value matrix converged after {iter} iterations!")
        break

V = Vs[iter] # Why is Python mad about this??
assert all(abs(v - 0.82352941) < 1e-5 for v in V[0:5]), "Value function is not calculated correctly"
assert all(abs(v - 0) < 1e-5 for v in V[[5,7,11,12,15]]), "Value function is not calculated correctly"

# Get the policy from this value matrix.
# Actually, Q should be up to date, so we can simply pick the best action from each state.
for state in range(n_state):
    policy_VI[state] = np.argmax(Q[state]) # Best action for this state

# Run it a couple times
scores = [run_episode(env, policy_VI, gamma, False) for _ in range(1000)]
print("VI Final score:", np.mean(scores))


# %% [markdown]

# Part 3: Questions
# 1. Policy iteration needs an inner loop to compute the value for a fixed policy. This inner loop does the same thing as the outer loop for value iteration: for each state/action pair, find the Q value of that state action pair. We need this inner loop in order to converge the value function.
# 2. Policy iteration takes 6 outer iterations. Value iteration takes 877 iterations to converge. These numbers are misleading, however, since at each of policy iteration's six steps it had to run the inner loop many times to converge the value function for that policy.
# 3. The Bellman optimality equation is greedy, calculating the Q function for all actions and always taking the highest valued action. We get a similar effect with policy iteration with the Bellman expectation equation, since the expected value of a policy that chooses the highest value IS the max (over a) of the Q function at that state.
# 4. Average score is from 0.72 to 0.8. This is because there is a region of FrozenLake where it is always possible to fall in a hole, no matter how careful you are.
# 5. Value iteration seems to be a lot faster for this problem, which has a very limited action space. However, if the action space were a lot larger, it would be difficult to generate Q-values for every state-action pair. In this case, a policy iteration method, which estimates the value of each state using an existing policy, would be less compute intensive.
# %%
