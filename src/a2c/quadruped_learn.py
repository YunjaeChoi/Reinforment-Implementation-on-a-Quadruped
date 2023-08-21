#/usr/bin/env python3
from __future__ import print_function
from quadruped_env import QuadrupedEnvironment
from a2c import A2C
import numpy as np

env = QuadrupedEnvironment()    #functions - init, jsp_callback, normalize_js, imu_sub_callback
                                # reset, step, 
state_shape = env.state_shape[0] #this returns tuple (34,) so we need to get only the first int
action_shape = env.action_shape[0] #this needs to be indexed too; same case as above
agent = A2C(state_shape,action_shape,actor_lr=0.001, critic_lr=0.001, gamma=0.99)

print('A2C agent configured') 
max_episode = 100
tot_rewards = []
print('env reset')
print("\n*************************")

observation, done = env.reset() #gazebo reset + states are 0 + join states are set to default
print("\n**********************")
print("obs, done? ",observation, done)
print("\n***********************")
action = agent.select_action(observation) # a random int action is taken, with prob = action_probs
print("Action after select_action(): ",action)
print("\n*********************")
observation, reward, done = env.step(action)

noise_sigma = 0.1
save_cutoff = 1
cutoff_count = 0
save_count = 0

curr_highest_eps_reward = -1000.0
for i in range(max_episode):
    if i % 100 == 0 and noise_sigma>0.03:
        agent.noise = 0.05 #constant noise for now 
        noise_sigma /= 2.0
    step_num = 0
    while done == False:
        step_num += 1
        state_val = env.step(action)[0]
        action_final = agent.select_action(state_val)
        print("Action after reset ",action_final)
        print("\n*********************")
        observation, reward, done = env.step(action_final[0])
        print('reward:',reward,'episode:', i, 'step:',step_num,'curr high eps reward:',curr_highest_eps_reward, 'saved:',save_count, 'cutoff count:', cutoff_count)
    action, eps_reward, done = env.step(action)
    tot_rewards.append(eps_reward)
    if eps_reward > curr_highest_eps_reward:
        cutoff_count += 1
        curr_highest_eps_reward = eps_reward
    if cutoff_count >= save_cutoff:
        save_count += 1
        print('saving_model at episode:',i)
        agent.save_model()
        agent.save_memory()
        cutoff_count = 0
    observation, done = env.reset()
np.save('eps_rewards',tot_rewards)

import matplotlib.pyplot as plt
plt.plot(tot_rewards)
