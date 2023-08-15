#!/usr/bin/ python3

import numpy as np
import tensorflow as tf

class A2C:
    def __init__(self, state_shape, action_shape, actor_lr=0.001, critic_lr=0.001, gamma=0.99,use_layer_norm=True):
        tf.compat.v1.reset_default_graph()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.nb_actions = np.prod(self.action_shape)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
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

    def set_model_loss(self, critic, actor_and_critic, actor_target, actor_and_critic_target, rewards, dones, gamma):
        Q_targets = rewards + (gamma * actor_and_critic_target) * (1. - dones)
        actor_loss = tf.reduce_mean(-actor_and_critic)
        tf.compat.v1.losses.add_loss(actor_loss)
        critic_loss = tf.compat.v1.losses.huber_loss(Q_targets,critic)
        return actor_loss, critic_loss

    def set_model_opt(self, actor_loss, critic_loss, actor_lr, critic_lr):
        train_vars = tf.compat.v1.trainable_variables()
        actor_vars = [var for var in train_vars if var.name.startswith('actor')]
        critic_vars = [var for var in train_vars if var.name.startswith('critic')]
        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            actor_opt = tf.compat.v1.train.AdamOptimizer(actor_lr).minimize(actor_loss, var_list=actor_vars)
            critic_opt = tf.compat.v1.train.AdamOptimizer(critic_lr).minimize(critic_loss, var_list=critic_vars)
        return actor_opt, critic_opt    
