from __future__ import print_function
from quadruped_env import QuadrupedEnvironment
from ddpg import DDPG


env = QuadrupedEnvironment()
state_shape = env.state_shape
action_shape = env.action_shape
agent = DDPG(state_shape,action_shape,batch_size=128,gamma=0.995,tau=0.001,
                                        actor_lr=0.0005, critic_lr=0.001, use_layer_norm=True)
print('DDPG agent configured')
agent.load_model(agent.current_path + '/model/model.ckpt')
max_episode = 5
for i in range(max_episode):
    print('env reset')
    observation, done = env.reset()
    action = agent.act_without_noise(observation)
    observation, reward, done = env.step(action)
    step_num = 0
    while done == False:
        step_num += 1
        action = agent.act_without_noise(observation)
        observation, reward, done = env.step(action)
        print('reward:',reward,'episode:', i, 'step:',step_num)
