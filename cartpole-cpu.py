import gym
import math
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple
import matplotlib.pyplot as plt

# Reference 1: https://www.youtube.com/watch?v=JNKvJEzuNsc 

# The original state space is continuous and this is a problem! Infinite number of states states!
# Divide the state space into discrete buckets so that the Q-table isn't infinitely large.
def q_table(env) -> np.ndarray:
    return np.zeros(N_BINS + (env.action_space.n, ))

# We only care about discretizing pole angle and pole velocity.
# We want to know if our polecart is falling down.
def discretizer(cartPosition, cartVelocity, poleAngle, poleVelocity) -> Tuple[int,...]:
    est = KBinsDiscretizer(n_bins=N_BINS, encode='ordinal', strategy='uniform')
    est.fit([LOWER_BOUNDS, UPPER_BOUNDS])
    return tuple(map(int, est.transform([[poleAngle, poleVelocity]])[0]))

def policy(qTable, state) -> int: # left or right
    return np.argmax(qTable[state])

def new_Q_value(qTable, reward, new_state, discount_factor=0.99) -> float:
    future_optimal_value = np.max(qTable[new_state])
    learning_value = reward + discount_factor * future_optimal_value
    return learning_value

# Hardcode a decaying learning rate
def learning_rate(episode, min_rate = 0.001) -> float:
    return max(min_rate, min(1.0, 1.0-math.log10((episode+1)/25)))

# Hardcode a decaying exploration rate
def exploration_rate(episode, min_rate = 0.01) -> float:
    return max(min_rate, min(1.0, 1.0-math.log10((episode+1)/25)))

def training(env, qTable, totalEpisodes=150) -> list[int]:

    samples = []
    for episode in range(totalEpisodes):

        if episode == totalEpisodes - 1: # Adjust according to what you want to record
            env = gym.wrappers.Monitor(env, "./videos", force=True, mode="training")

        done = False
        current_state = discretizer(*env.reset())
        i = 0
        while not done:
            if np.random.random() < exploration_rate(episode):
                action = env.action_space.sample()
            else:
                action = policy(qTable, current_state)

            observation, reward, done, _ = env.step(action)
            new_state = discretizer(*observation)

            lr = learning_rate(episode)
            learned_value = new_Q_value(qTable, reward, new_state)
            old_value = qTable[current_state][action]
            qTable[current_state][action] = (1-lr) * old_value + lr*learned_value

            current_state = new_state
            i += 1
        samples.append(i)

        if ((episode + 1) % 50) == 0:
            print(f"Finished training episode: {episode + 1}")
    
    return samples

env = gym.make("CartPole-v0")
env.reset()
LOWER_BOUNDS = [env.observation_space.low[2], -math.radians(50)]
UPPER_BOUNDS = [env.observation_space.high[2], math.radians(50)]
N_BINS = (6,12) # Discretize pole angle into 6 buckets, and pole velocity into 12 buckets.

# Initiate Q-table and run training
qTable = q_table(env)
print(qTable.shape)
samples = training(env, qTable)

# Plot how long the cartpole stays alive
plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.plot(samples, color='blue', alpha=0.7)
plt.title('Training using Q-Learning')
plt.ylabel('Frames Alive')
plt.xlabel('Training Episodes')
plt.show()
