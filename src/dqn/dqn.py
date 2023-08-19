import numpy as np
import tensorflow as tf
import time
import os
import pickle

class ReplayBuffer:

    def __init__(self, maxlen, action_shape, state_shape, dtype=np.float32):
        # Initialize a ReplayBuffer object
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.state_data = np.zeros((maxlen,) + state_shape).astype(dtype)
        self.action_data = np.zeros((maxlen,) + action_shape).astype(dtype)
        self.reward_data = np.zeros((maxlen,1)).astype(dtype)
        self.next_state_data = np.zeros((maxlen,) + state_shape).astype(dtype)
        self.done_data = np.zeros((maxlen,1)).astype(dtype)

    def add(self, state, action, reward, next_state, done):
        # Add a new experience to memory
        if self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen
        else:
            self.length += 1
        idx = (self.start + self.length - 1) % self.maxlen
        self.state_data[idx] = state
        self.action_data[idx] = action
        self.reward_data[idx] = reward
        self.next_state_data[idx] = next_state
        self.done_data[idx] = done

    def sample(self, batch_size=64):
        # Randomly sample a batch of experiences from memory
        idxs = np.random.randint(0,self.length - 1, size=batch_size)
        sampled = {'states':self.set_min_ndim(self.state_data[idxs]),
                   'actions':self.set_min_ndim(self.action_data[idxs]),
                   'rewards':self.set_min_ndim(self.reward_data[idxs]),
                   'next_states':self.set_min_ndim(self.next_state_data[idxs]),
                   'dones':self.set_min_ndim(self.done_data[idxs])}
        return sampled

    def set_min_ndim(self,x):
        # set numpy array minimum dim to 2 (for sampling)
        if x.ndim < 2:
            return x.reshape(-1,1)
        else:
            return x

    def __len__(self):
        return self.length

class OUNoise:
    # Ornstein-Uhlenbeck process of noise insertion
    def __init__(self, size, mu=None, theta=0.15, sigma=0.03, dt=1e-2):
        # Initialize parameters and noise process
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        # Reset the internal state (= noise) to mean (mu)
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        # Update internal state and return it as a noise sample
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(len(x))
        self.state = x + dx
        return self.state


# a DQN class 
class DQN:

    # Reinforcement Learning agent that learns using deep Q learning
    def __init__(self,state_shape,action_shape,batch_size=128,gamma=0.995,tau=0.005, actor_lr=0.0001, critic_lr=0.001,use_layer_norm=True):
        
         # Algorithm parameters
        self.gamma = gamma # discount factor
        self.tau = tau # soft update
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        tf.compat.v1.reset_default_graph()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.nb_actions = np.prod(self.action_shape)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.use_layer_norm = use_layer_norm

        # inputs
        tf.compat.v1.disable_eager_execution()
        self.input_state = tf.compat.v1.placeholder(tf.float32, (None,) + self.state_shape, name='input_state')
        self.input_action = tf.compat.v1.placeholder(tf.float32, (None,) + self.action_shape, name='input_action')
        self.input_state_target = tf.compat.v1.placeholder(tf.float32, (None,) + self.state_shape, name='input_state_target')
        self.rewards = tf.compat.v1.placeholder(tf.float32, (None,1), name='rewards')
        self.dones =tf.compat.v1.placeholder(tf.float32, (None,1), name='dones')

        # local and target nets
        self.actor = self.actor_net(self.input_state, self.nb_actions,name='actor',use_layer_norm=self.use_layer_norm)
        self.critic = self.critic_net(self.input_state, self.input_action,name='critic',use_layer_norm=self.use_layer_norm)
        self.actor_and_critic = self.critic_net(self.input_state,self.actor,name='critic',reuse=True,use_layer_norm=self.use_layer_norm)

        self.actor_target = self.actor_net(self.input_state_target, self.nb_actions, name='target_actor',use_layer_norm=self.use_layer_norm)
        self.actor_and_critic_target = self.critic_net(self.input_state_target,
                                                       self.actor_target, name='target_critic',use_layer_norm=self.use_layer_norm)

        self.actor_loss, self.critic_loss = self.set_model_loss(self.critic, self.actor_and_critic,
                                                                self.actor_target, self.actor_and_critic_target,
                                                                self.rewards, self.dones, self.gamma)

        self.actor_opt, self.critic_opt = self.set_model_opt(self.actor_loss, self.critic_loss,
                                                             self.actor_lr, self.critic_lr)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.buffer_size,self.action_shape, self.state_shape)

        # Noise process
        self.noise = OUNoise(self.nb_actions)

        # initialize
        self.initialize()
        self.saver = tf.compat.v1.train.Saver()
        self.current_path = os.getcwd()

        # initial episode vars
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0
        self.episode_num = 0
        
    # actor network
    def actor_net(self, state, nb_actions, name, reuse=False, training=True, use_layer_norm=True):
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            x = tf.keras.layers.Dense(130)(state)
            if use_layer_norm:
                # x = tf.contrib.layers.layer_norm(x)
                layer_norma = tf.keras.layers.LayerNormalization(axis = -1)
                x = layer_norma(x)
            x = tf.nn.relu(x)
            x = tf.keras.layers.Dense(100)(x)
            if use_layer_norm:
                # x = tf.contrib.layers.layer_norm(x)
                layer_norma = tf.keras.layers.LayerNormalization(axis = -1)
                x = layer_norma(x)
            x = tf.nn.relu(x)
            x = tf.keras.layers.Dense(80)(x)
            if use_layer_norm:
                # x = tf.contrib.layers.layer_norm(x)
                layer_norma = tf.keras.layers.LayerNormalization(axis = -1)
                x = layer_norma(x)
            x = tf.nn.relu(x)
            actions = tf.keras.layers.Dense(nb_actions, activation=tf.tanh, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(x)
            return actions
    
    # critic network
    def critic_net(self, state, action, name, reuse=False, training=True, use_layer_norm=True):
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            x = tf.keras.layers.Dense(130)(state)
            if use_layer_norm:
                # x = tf.contrib.layers.layer_norm(x)
                layer_norma = tf.keras.layers.LayerNormalization(axis = -1)
                x = layer_norma(x)
            x = tf.nn.relu(x)
            x = tf.concat([x, action], axis=-1)
            x = tf.keras.layers.Dense(100)(x)
            if use_layer_norm:
                # x = tf.contrib.layers.layer_norm(x)
                layer_norma = tf.keras.layers.LayerNormalization(axis = -1)
                x = layer_norma(x)
            x = tf.nn.relu(x)
            x = tf.keras.layers.Dense(80)(x)
            if use_layer_norm:
                # x = tf.contrib.layers.layer_norm(x)
                layer_norma = tf.keras.layers.LayerNormalization(axis = -1)
                x = layer_norma(x)
            x = tf.nn.relu(x)
            q = tf.keras.layers.Dense(1,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(x)
            return q

    # helper to set loss function
    def set_model_loss(self, critic, actor_and_critic, actor_target, actor_and_critic_target, rewards, dones, gamma):
        Q_targets = rewards + (gamma * actor_and_critic_target) * (1. - dones)
        actor_loss = tf.reduce_mean(-actor_and_critic)
        tf.compat.v1.losses.add_loss(actor_loss)
        critic_loss = tf.compat.v1.losses.huber_loss(Q_targets,critic)
        return actor_loss, critic_loss

    # helper to set optimizer 
    def set_model_opt(self, actor_loss, critic_loss, actor_lr, critic_lr):
        train_vars = tf.compat.v1.trainable_variables()
        actor_vars = [var for var in train_vars if var.name.startswith('actor')]
        critic_vars = [var for var in train_vars if var.name.startswith('critic')]
        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            actor_opt = tf.compat.v1.train.AdamOptimizer(actor_lr).minimize(actor_loss, var_list=actor_vars)
            critic_opt = tf.compat.v1.train.AdamOptimizer(critic_lr).minimize(critic_loss, var_list=critic_vars)
        return actor_opt, critic_opt    

    # reset episode variables
    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0

    # step function for internaliing learned parameters after each episode
    def step(self, state, reward, done):
        action = self.act(state)
        self.count += 1
        if self.last_state is not None and self.last_action is not None:
            self.total_reward += reward
            self.memory.add(self.last_state, self.last_action, reward, state, done)
        if (len(self.memory) > self.batch_size):
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
        self.last_state = state
        self.last_action = action
        if done:
            self.episode_num += 1
            eps_reward = self.total_reward
            print('Episode {}: total reward={:7.4f}, count={}'.format(self.episode_num,self.total_reward,self.count))
            self.reset_episode_vars()
            return action, eps_reward
        else:
            return action

    def act(self, states):
        # Returns actions for given state(s) as per current policy
        actions = self.sess.run(self.actor, feed_dict={self.input_state:states})
        noise = self.noise.sample()
        print('noise:',noise)
        return np.clip(actions + noise,a_min=-1.,a_max=1.).reshape(self.action_shape)

    def act_without_noise(self, states):
        # Returns actions for given state(s) as per current policy
        actions = self.sess.run(self.actor, feed_dict={self.input_state:states})
        return np.array(actions).reshape(self.action_shape)

    def learn(self, experiences):
        # Update policy and value parameters using given batch of experience tuples
        states = experiences['states']
        actions = experiences['actions']
        rewards = experiences['rewards']
        next_states = experiences['next_states']
        dones = experiences['dones']

        # actor critic update
        self.sess.run([self.actor_opt,self.critic_opt],feed_dict={self.input_state:states,
                                                                              self.input_action:actions,
                                                                              self.input_state_target:next_states,
                                                                              self.rewards:rewards,
                                                                              self.dones:dones})
        # target soft update
        self.sess.run(self.soft_update_ops)

    # global a2c model initialization variables 
    def initialize(self):
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        actor_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        actor_target_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor')
        critic_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        critic_target_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic')
        target_init_ops = []
        soft_update_ops = []
        for var, target_var in zip(actor_var, actor_target_var):
            target_init_ops.append(tf.compat.v1.assign(target_var,var))
            soft_update_ops.append(tf.compat.v1.assign(target_var, (1. - self.tau) * target_var + self.tau * var))
        for var, target_var in zip(critic_var, critic_target_var):
            target_init_ops.append(tf.compat.v1.assign(target_var,var))
            soft_update_ops.append(tf.compat.v1.assign(target_var, (1. - self.tau) * target_var + self.tau * var))
        self.soft_update_ops = soft_update_ops
        self.sess.run(target_init_ops)

    # save model weights
    def save_model(self):
        self.saver.save(self.sess,self.current_path + '/model/model.ckpt')

    # load model weights
    def load_model(self,path):
        self.saver.restore(self.sess,path)

    # save agent memory
    def save_memory(self):
        mem_file = open(self.current_path + '/agent_mem.p','wb')
        pickle.dump(self.memory,mem_file)
        mem_file.close()

    # load agent memory
    def load_memory(self,path):
        mem_file = open(self.current_path + '/agent_mem.p','rb')
        mem = pickle.load(mem_file)
        self.memory = mem
        mem_file.close()
