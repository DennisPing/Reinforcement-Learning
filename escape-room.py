from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import matplotlib.pyplot as plt
import math

# Reference 1: https://www.youtube.com/watch?v=JNKvJEzuNsc 
# Reference 2: https://www.youtube.com/watch?v=bD6V3rcr_54

class MazeEnv(Env):
    def __init__(self):
        self.action_space = Discrete(6) # 6 possible actions: 0,1,2,3,4,5
        self.maze = {
            0: [4],
            1: [3, 5],
            2: [3],
            3: [1, 2, 4],
            4: [0, 3, 5],
            5: [1, 4]
        }
        self.observation_space = Box(low=0, high=5, shape=(self.action_space.n, ))
        self.state = 2 # starting room
        self.save_render = False
        self.frames = []

    def step(self, action):
        # If the action is a valid room, reward the AI. Else penalize the AI.
        if action in self.maze[self.state]:
            self.state = action
            reward = 1
            if self.save_render:
                self.frames.append(action)
        else:
            reward = -1

        # Now check if you reached the end goal
        if self.state == 5:
            done = True
        else:
            done = False

        info = {}
        return self.state, reward, done, info

    def render(self):
        print(f"Actions to take: {self.frames}")

    def reset(self):
        self.state = 2
        return self.state

def policy(qTable, state) -> int:
    return np.argmax(qTable[state])

def new_Q_value(qTable, reward, new_state, discount_factor=0.95) -> float:
    future_optimal_value = np.max(qTable[new_state])
    learning_value = reward + discount_factor * future_optimal_value
    return learning_value

# Hardcode a decaying learning rate
def learning_rate(episode, min_rate = 0.001) -> float:
    return max(min_rate, min(1.0, 1.0-math.log10((episode+1)/25)))

# Hardcode a decaying exploration rate
def exploration_rate(episode, min_rate = 0.01) -> float:
    return max(min_rate, min(1.0, 1.0-math.log10((episode+1)/25)))

def training(env, qTable, totalEpisodes=300) -> list[int]:
    samples = []
    for episode in range(totalEpisodes):

        if episode == totalEpisodes - 1:
            env.save_render = True # Save only the last episode
        
        done = False
        current_state = env.reset()
        i = 0
        while not done:
            if np.random.random() < exploration_rate(episode):
                action = env.action_space.sample()
            else:
                action = policy(qTable, current_state)

            observation, reward, done, _ = env.step(action)
            new_state = observation

            lr = learning_rate(episode)
            learned_value = new_Q_value(qTable, reward, new_state)
            old_value = qTable[current_state, action]
            qTable[current_state][action] = ((1 - lr) * old_value) + (lr * learned_value)

            current_state = new_state
            i += 1
        samples.append(i)

        if ((episode + 1) % 50) == 0:
            print(f"Finished training episode: {episode + 1}")
    return samples

env = MazeEnv()
qTable = np.array([[-1, -1, -1, -1, 0, -1],
                    [-1, -1, -1, 0, -1, 100],
                    [-1, -1, -1, 0, -1, -1],
                    [-1, 0, 0, -1, 0, -1],
                    [0, -1, -1, 0, -1, 100],
                    [-1, 0, -1, -1, 0, 100]])
samples = training(env, qTable)
env.render() # Render the last episode
print(f"Lowest number of actions to complete maze: {samples[-1]}")

plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.plot(samples, color='blue', alpha=0.7)
plt.title('Training using Q-Learning')
plt.ylabel('Number of Actions to Reach Goal')
plt.xlabel('Training Episodes')
plt.show()
