# Lab 4: Monte Carlo Methods

## Overview
This lab introduces Monte Carlo (MC) methods for model-free RL. Unlike Dynamic Programming, MC methods learn directly from experience without requiring a model of the environment. You will implement MC prediction and MC control.

## Objectives
- Understand the difference between model-based and model-free RL
- Implement Monte Carlo Prediction (first-visit and every-visit)
- Implement Monte Carlo Control with Exploring Starts
- Explore on-policy and off-policy MC methods

## Instructions
1. Implement first-visit and every-visit MC Prediction for a simple environment
2. Implement MC Control to find an optimal policy
3. Compare on-policy and off-policy MC methods

## Files
- `lab4.py` — Starter code for the lab

## Deliverables
- Completed `lab4.py`
- Short write-up answering the reflection questions below

## Reflection Questions
1. What is the difference between first-visit and every-visit MC Prediction?
2. Why do MC methods require complete episodes before updating estimates?
3. What is the importance sampling ratio in off-policy MC methods?
