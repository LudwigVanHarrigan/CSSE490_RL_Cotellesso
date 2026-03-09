# Lab 7: Policy Gradient Methods

## Overview
This lab covers Policy Gradient methods, a family of RL algorithms that directly optimize the policy rather than learning a value function. You will implement the REINFORCE algorithm and explore actor-critic architectures.

## Objectives
- Understand the policy gradient theorem
- Implement the REINFORCE (Monte Carlo Policy Gradient) algorithm
- Implement a basic Actor-Critic algorithm
- Compare policy gradient methods to value-based methods from previous labs

## Instructions
1. Implement the REINFORCE algorithm with a neural network policy
2. Add a baseline to reduce variance in the REINFORCE updates
3. Implement a simple Actor-Critic method
4. Evaluate and compare the performance of REINFORCE and Actor-Critic

## Files
- `lab7.py` — Starter code for the lab

## Deliverables
- Completed `lab7.py`
- Training curves (reward vs. episodes)
- Short write-up answering the reflection questions below

## Reflection Questions
1. What is the policy gradient theorem and why is it useful?
2. How does adding a baseline help reduce variance? What makes a good baseline?
3. What are the trade-offs between policy gradient methods and value-based methods like DQN?
