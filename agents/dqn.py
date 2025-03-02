# -- Public Imports
import os
import random
import pickle
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, Reshape, BatchNormalization, Flatten
from tensorflow.keras.models import Model

# -- Private Imports
from utils import *
from constants import *

# -- Global Variables

# -- Functions

# Base Agent for DQN
class BaseAgentDQN:
    def __init__(self, config):
        self.state_space = config.state_space
        self.action_space = config.action_space
        self.action_mapper = ActionMapper(minVal=0, maxVal=self.action_space)

        self.agent_type = config.agent

        # Buffer
        self.buffer_capacity = config.buffer_capacity if hasattr(config, 'buffer_capacity') else int(1e4)
        self.batch_size = config.batch_size if hasattr(config, 'batch_size') else 128
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, self.state_space))
        self.action_buffer = np.zeros((self.buffer_capacity, 1))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.state_space))

        # Hyper-parameters
        self.epsilon = config.epsilon
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.gamma = config.gamma  # Discount factor
        self.learning_rate = config.dqn_lr  # Learning rate for the DQN network
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_func = tf.keras.losses.Huber()
        # self.loss_func = tf.keras.losses.MeanSquaredError()

        # Create Deep Q Network
        self.model = self.create_model()
        self.target_model = self.create_model()
        print(self.model.summary())

    def create_model(self):
        input_shape = (self.state_space, 1)  # Add a channel dimension (1) for the CNN

        # Input Layer
        X_input = Input(input_shape)

        # 1D Convolutional Layer
        X = Conv1D(filters=32, kernel_size=3, activation="relu")(X_input)
        X = Conv1D(filters=64, kernel_size=3, activation="relu")(X)
        X = Conv1D(filters=128, kernel_size=3, activation="relu")(X)

        # Flatten the output of the CNN to feed into Dense layers
        X = Flatten()(X)

        # Dense Layers
        X = Dense(512, activation="relu")(X)
        X = Dense(512, activation="relu")(X)
        X = Dense(256, activation="relu")(X)

        # Output Layer (action space size)
        output = Dense(self.action_space, activation="linear")(X)

        # Create Model
        model = Model(inputs=X_input, outputs=output)

        return model

    def record(self, obs_tuple):
        assert len(obs_tuple) == 4

        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    def act(self, state):
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        if state.ndim==1:
            state = np.expand_dims(state, axis=0)

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.random() < self.epsilon:
            action_idx = np.random.choice(self.action_space)
        else:
            q_vals_dist = self.model.predict(state, verbose=0)[0]
            action_idx = tf.argmax(q_vals_dist).numpy()

        action = self.action_mapper.idx_to_bool_action(action_idx)
        return action, action_idx

    def sample(self):
        sample_indices = np.random.choice(min(self.buffer_counter, self.buffer_capacity), self.batch_size)

        state_sample = tf.convert_to_tensor(self.state_buffer[sample_indices])
        action_sample = tf.convert_to_tensor(self.action_buffer[sample_indices])
        reward_sample = tf.cast(tf.convert_to_tensor(self.reward_buffer[sample_indices]), dtype=tf.float32)
        next_state_sample = tf.convert_to_tensor(self.next_state_buffer[sample_indices])

        return state_sample, action_sample, reward_sample, next_state_sample

    @tf.function
    def update(self):
        state_sample, action_sample, reward_sample, next_state_sample = self.sample()
        action_sample_int = tf.cast(tf.squeeze(action_sample), tf.int32)

        target_q_vals = tf.reduce_max(self.target_model(next_state_sample), axis=1)
        y = reward_sample + tf.expand_dims(self.gamma * target_q_vals, axis=1)
        mask = tf.one_hot(action_sample_int, self.action_space)

        with tf.GradientTape() as tape:
            q_vals = self.model(state_sample)

            q_action = tf.reduce_sum(tf.multiply(q_vals, mask), axis=1)

            loss = self.loss_func(y, q_action)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    @tf.function
    def update_target(self, tau=0.001):
        # Update target actor
        # self.target_model.set_weights(self.model.get_weights())
        for (a, b) in zip(self.target_model.variables, self.model.variables):
            a.assign(b * tau + (1 - tau))

    def save_model(self):

        file_path = f"./models/dqn/model_dqn.pkl"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the model and optimizer state
        model_data = {
            'model_architecture': self.model.get_config(),
            'model_weights': self.model.get_weights(),
        }

        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {file_path}")

    def load_model(self):
        file_path = f"./models/dqn/model_dqn.pkl"
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = Model.from_config(model_data.get('model_architecture'))
        self.model.set_weights(model_data.get('model_weights'))

        print(f"Model loaded from {file_path}")