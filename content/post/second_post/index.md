+++
date = '2025-10-13T17:22:29-04:00'
draft = false
title = 'RL report'
tags = ["AI", "NLP", "Deep Learning"]
categories = ["Technology"]
+++

{{< mathjax >}}

# Reinforcement Learning Experiment Report – Session 2

Name: **Yang Zi’ang**
Student ID: **21307181**

### I. Experiment Title

Implementing Q-learning and SARSA algorithms based on the *Cliff Walk* example

### II. Experiment Content

#### 1. Q-learning Algorithm Principle

**Q-learning Algorithm**

Q-learning is a **model-free reinforcement learning algorithm** that approximates the optimal value function by updating the state–action value function $Q(s, a)$. It is an **off-policy** method because the actions used for updates are not necessarily those taken by the current policy.

**Formula and Update Rule**

The Q-learning update rule is as follows:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$

* $Q(s, a)$: Value of taking action $a$ in state $s$.
* $\alpha$: Learning rate, controlling the update step size ($0 < \alpha \leq 1$).
* $\gamma$: Discount factor, measuring the importance of future rewards ($0 \leq \gamma \leq 1$).
* $r_{t+1}$: Immediate reward received after the current step.
* $\max_{a'} Q(s_{t+1}, a')$: Maximum action value for the next state, representing a greedy policy.

**Pseudocode**

1. Initialize $Q(s, a)$, usually to zero.
2. For each episode:

   * Initialize the starting state $s_0$.
   * Choose an action $a_t$ according to the $\epsilon$-greedy policy.
   * Execute action $a_t$, obtain reward $r_{t+1}$ and next state $s_{t+1}$.
   * Update $Q(s_t, a_t)$ using the update formula.
   * Set $s_t \leftarrow s_{t+1}$.
3. Repeat until the termination condition is met.

#### 2. SARSA Algorithm Principle

**SARSA Algorithm**

SARSA is also a **model-free reinforcement learning algorithm**, similar to Q-learning, but it is an **on-policy** method. The value function is updated based on the action actually taken by the current policy.

**Formula and Update Rule**

The SARSA update rule is as follows:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]
$$

* $Q(s, a)$: Value of taking action $a$ in state $s$.
* $a_{t+1}$: Next action chosen under the current policy in state $s_{t+1}$.
* Other symbols have the same meanings as in Q-learning.

**Pseudocode**

1. Initialize $Q(s, a)$, usually to zero.
2. For each episode:

   * Initialize the starting state $s_0$.
   * Choose the initial action $a_0$ according to the $\epsilon$-greedy policy.
   * For each step:

     * Execute action $a_t$, obtain reward $r_{t+1}$ and next state $s_{t+1}$.
     * Choose the next action $a_{t+1}$ according to the current policy.
     * Update $Q(s_t, a_t)$ using the update formula.
     * Set $s_t \leftarrow s_{t+1}$, $a_t \leftarrow a_{t+1}$.
3. Repeat until the termination condition is met.

#### 3. Comparison Between Algorithms

| Feature              | Q-learning                                                    | SARSA                                                           |
| -------------------- | ------------------------------------------------------------- | --------------------------------------------------------------- |
| Policy Type          | Off-policy                                                    | On-policy                                                       |
| Update Method        | Based on greedy action$\max Q$                              | Based on action chosen by the policy$Q$                       |
| Convergence Property | Less sensitive to exploration, easier to reach optimal policy | More stable convergence, but strongly affected by policy choice |

---
