import gym
import numpy as np
import random
import tensorflow as tf
from time import sleep
import os

def print_frames(frames):
    for i, frame in enumerate(frames):
        os.system('clear')
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(1)


# Init Taxi-V2 Env and load q_table
env = gym.make("Taxi-v2").env
q_table=np.load("q_table_2layer_new.npy")

########################################################################

"""Evaluate agent's performance after Q-learning"""

all_epochs, all_penalties = 0, 0

for _ in range(10):
    total_epochs, total_penalties = 0, 0
    episodes = 100
    for _ in range(episodes):
        state = env.reset()
        
        env.render()
        epochs, penalties, reward = 0, 0, 0
        done = False
        frames = []
        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1


            frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
                })

            epochs += 1

        total_penalties += penalties
        total_epochs += epochs

    print_frames(frames)
    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")
    sleep(3)

    all_penalties += total_penalties
    all_epochs += total_epochs

print(f"Average timesteps per episode: {all_epochs / total_epochs}")

