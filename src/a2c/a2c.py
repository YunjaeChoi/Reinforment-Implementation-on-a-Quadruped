#!/usr/bin/ python3

#this is the actor critic model which is used for the standalone implementation of a2c. The one on ddpg is different

import tensorflow as tf
import numpy as np
import pickle
import os

class A2C:
    
    def __init__(self, state_dim, action_dim, actor_lr=0.9, critic_lr=0.9, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma

        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.current_path = os.getcwd()

    def build_actor(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.actor_lr), loss='categorical_crossentropy')
        return model

    def build_critic(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.critic_lr), loss='mean_squared_error')
        return model

    def select_action(self, state):

        print ("State input to 'select_action()': ",state)
        print("\n************************************")
        # Sample an action from the actor's policy distribution
        action_probs = self.actor.predict(np.array(state))[0]
        return action_probs

    def train(self, state, action, reward, next_state, done):
        state = np.array([state])
        next_state = np.array([next_state])

        # Calculate TD target for the critic
        if done:
            target = reward
        else:
            next_value = self.critic.predict(next_state)[0][0]
            target = reward + self.gamma * next_value

        # Calculate advantage
        current_value = self.critic.predict(state)[0][0]
        advantage = target - current_value

        # Update actor and critic
        with tf.GradientTape() as tape:
            action_probs = self.actor(state)
            selected_action_prob = action_probs[0, action]
            actor_loss = -tf.math.log(selected_action_prob) * advantage

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        self.critic.fit(state, target, verbose=0)
        
    # def initialize(self):
    #     self.sess = tf.compat.v1.Session()
    #     self.sess.run(tf.compat.v1.global_variables_initializer())
    #     actor_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
    #     actor_target_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor')
    #     critic_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
    #     critic_target_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic')
    #     target_init_ops = []
    #     soft_update_ops = []
    #     for var, target_var in zip(actor_var, actor_target_var):
    #         target_init_ops.append(tf.compat.v1.assign(target_var,var))
    #         soft_update_ops.append(tf.compat.v1.assign(target_var, (1. - self.tau) * target_var + self.tau * var))
    #     for var, target_var in zip(critic_var, critic_target_var):
    #         target_init_ops.append(tf.compat.v1.assign(target_var,var))
    #         soft_update_ops.append(tf.compat.v1.assign(target_var, (1. - self.tau) * target_var + self.tau * var))
    #     self.soft_update_ops = soft_update_ops
    #     self.sess.run(target_init_ops)

    # def save_model(self):
    #     self.saver.save(self.sess,self.current_path + '/model/model.ckpt')

    # def load_model(self,path):
    #     self.saver.restore(self.sess,path)

    # def save_memory(self):
    #     mem_file = open(self.current_path + '/agent_mem.p','wb')
    #     pickle.dump(self.memory,mem_file)
    #     mem_file.close()

    # def load_memory(self,path):
    #     mem_file = open(self.current_path + '/agent_mem.p','rb')
    #     mem = pickle.load(mem_file)
    #     self.memory = mem
    #     mem_file.close()