#!/usr/bin/ python3

import tensorflow as tf
import numpy as np

class A2C:
    
    def __init__(self, state_dim, action_dim, actor_lr=0.001, critic_lr=0.005, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma

        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor(self):
        print("\n*******************")
        print("action_shape : ",self.action_dim)
        print("state_shape : ",self.state_dim)
        print("\n*******************")
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

        print ("State : ",state)
        print("\n************************************")
        # Sample an action from the actor's policy distribution
        action_probs = self.actor.predict(np.array(state))[0]
        action_choice = np.random.choice(self.action_dim, p=action_probs)
        print("Action after select_action() ",action_choice)
        print("\n*************************************")
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

    def save_models(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

    def load_models(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
