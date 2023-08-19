# uses the trained weight files for deploying the quadruped hardware

from __future__ import print_function
from quadruped_model import QuadrupedEnvironment
from ddpg import DDPG
import time
import numpy as np

# declaring the environment
env = QuadrupedEnvironment()

# model parameters
state_shape = env.state_shape
action_shape = env.action_shape

# model configuration
agent = DDPG(state_shape,action_shape,batch_size=128,gamma=0.995,tau=0.001, actor_lr=0.0005, critic_lr=0.001, use_layer_norm=True)
print('DDPG agent configured')

# loading the model
agent.load_model(agent.current_path + '/model/model.ckpt')
print('Resetting joint positions')
observation = env.reset()
print('Reset')
time.sleep(1.0)

# running for 25 iterations
for i in range(25):
    action = agent.act_without_noise(observation)
    observation = env.step(action)
    #time.sleep(0.25)

# resetting walk
print('Resetting joint positions')
#env.reset()
print('Reset')
