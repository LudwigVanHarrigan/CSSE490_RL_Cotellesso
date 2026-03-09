# CSSE490_RL_Cotellesso

A course repository for CSSE490: Reinforcement Learning. This repository contains seven labs that progress from foundational RL concepts through deep learning-based approaches.

## Repository Structure

```
CSSE490_RL_Cotellesso/
├── lab1/   # Introduction to Reinforcement Learning
├── lab2/   # Markov Decision Processes (MDPs)
├── lab3/   # Dynamic Programming
├── lab4/   # Monte Carlo Methods
├── lab5/   # Temporal Difference Learning
├── lab6/   # Deep Q-Networks (DQN)
└── lab7/   # Policy Gradient Methods
```

## Labs

### [Lab 1: Introduction to Reinforcement Learning](lab1/README.md)
Set up your RL development environment and explore the core components of an RL system — agents, environments, states, actions, and rewards — by implementing a simple random agent.

### [Lab 2: Markov Decision Processes (MDPs)](lab2/README.md)
Formalize RL problems as MDPs. Implement a grid-world environment, define policies, and compute expected returns under different discount factors.

### [Lab 3: Dynamic Programming](lab3/README.md)
Solve MDPs with full model knowledge using Policy Evaluation, Policy Iteration, and Value Iteration. Compare convergence behavior across algorithms.

### [Lab 4: Monte Carlo Methods](lab4/README.md)
Learn directly from experience without a model. Implement first-visit and every-visit Monte Carlo Prediction as well as MC Control with on-policy and off-policy variants.

### [Lab 5: Temporal Difference Learning](lab5/README.md)
Combine DP and MC ideas with online, incremental updates. Implement SARSA (on-policy) and Q-Learning (off-policy) and compare their behaviors on a control task.

### [Lab 6: Deep Q-Networks (DQN)](lab6/README.md)
Scale Q-Learning to high-dimensional state spaces using deep neural networks. Implement experience replay and a target network, then train a DQN agent and analyze its learning curves.

### [Lab 7: Policy Gradient Methods](lab7/README.md)
Directly optimize the policy using gradient ascent. Implement the REINFORCE algorithm, add a variance-reducing baseline, and explore an Actor-Critic architecture.

## Getting Started

1. Clone this repository
2. Install dependencies (see individual lab READMEs for specific requirements)
3. Navigate to the lab folder and follow the instructions in its `README.md`
