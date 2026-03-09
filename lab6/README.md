# Lab 6: Deep Q-Networks (DQN)

## Overview
This lab introduces Deep Q-Networks (DQN), which combine Q-Learning with deep neural networks to handle high-dimensional state spaces. You will implement a DQN agent and train it on a classic control or Atari environment.

## Objectives
- Understand how neural networks can be used to approximate Q-values
- Implement the DQN algorithm including experience replay and a target network
- Train a DQN agent on a chosen environment
- Analyze training stability and the role of key hyperparameters

## Instructions
1. Set up a neural network to approximate the Q-function
2. Implement experience replay (replay buffer)
3. Implement the target network and soft/hard update strategies
4. Train the DQN agent and plot learning curves

## Files
- `lab6.py` — Starter code for the lab

## Deliverables
- Completed `lab6.py`
- Training curves (reward vs. episodes)
- Short write-up answering the reflection questions below

## Reflection Questions
1. Why is experience replay important in DQN?
2. What problem does the target network address?
3. How do hyperparameters such as learning rate and replay buffer size affect performance?
