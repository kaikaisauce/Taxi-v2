import gym
import numpy as np
import random
import tensorflow as tf
from time import sleep
import os
from tensorflow.python.framework import ops


def print_frames(frames):
    for i, frame in enumerate(frames):
        os.system('clear')
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        #sleep()


# Init Taxi-V2 Env
env = gym.make("Taxi-v2").env
q_table = np.zeros([env.observation_space.n, env.action_space.n])


def inference(inputs1):
    W = tf.Variable(tf.random_uniform([500,6],-1.01,-1))
    Qout = tf.matmul(inputs1,W)
    predict = tf.argmax(Qout,1)
    return predict, Qout, W

#######################################################################

def loss(nextQ, Qout):
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    return loss

#######################################################################

def train(loss):
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)
    return updateModel 

######################################################################

def evaluate(nextQ, Qout):
    correct_prediction = Qout
    float_val = tf.cast(correct_prediction, tf.float32)
    prediction_as_float = tf.reduce_mean(float_val)
    print (prediction_as_float)

######################################################################

ops.reset_default_graph()


inputs1 = tf.placeholder(shape=[1,500],dtype=tf.float32)
nextQ = tf.placeholder(shape=[6,],dtype=tf.float32)

#####################################################################

predict, Qout, W = inference(inputs1)
cost = loss(nextQ, Qout)
trainOp = train(cost)


######################################################################

init = tf.initialize_all_variables()
alpha = 0.1
gamma = 0.9
epsilon=0.1

######################################################################

#create lists to contain total rewards and steps per episode
all_epochs = []
all_penalties = []

######################################################################

with tf.Session() as sess:
    sess.run(init)
    for i in range(1,100001):
        state = env.reset()   
        epochs, penalties, reward, = 0, 0, 0
        done = False
        frames = []
        
        while not done:
            if random.uniform(0, 1) < epsilon:
            # Check the action space
                action = env.action_space.sample()
            else:
            # Check the learned values
                action,q_table[state] = sess.run([predict,Qout],feed_dict={inputs1:np.identity(500)[state:state+1]})
                
            #print("Action::::",action)
            if str(type(action)) == "<class 'numpy.ndarray'>":
                action = action[0]

            next_state, reward, done, info = env.step(action)
            #print (q_table[state])
            
            old_value = q_table[state, action]
            #next_max = np.max(q_table[next_state])

            next_max = np.max(sess.run(Qout,feed_dict={inputs1:np.identity(500)[next_state:next_state+1]}))
            #print (maxQ1)
            new_value = (1 - alpha) * old_value + alpha * \
                (reward + gamma * next_max)
            q_table[state, action] = new_value
        # Update the new value
            #print (q_table[state])
            info,W1 = sess.run([trainOp, W],feed_dict={inputs1:np.identity(500)[state:state+1],nextQ:q_table[state]})
            
            if reward == -10:
                penalties += 1

            frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
                })

            state = next_state
            print_frames(frames)
            epochs += 1
            print("epochs: ",epochs)
            print("penalties: ",penalties)

        if i % 100 == 0:
            #os.system('clear')
            print("Episode: {}".format(i))


    print("Training finished.")

####################################################3
###save q table#############

np.save("q_table.npy",q_table)

q_table=np.load("q_table.npy")

########################################################################

"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 1


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


