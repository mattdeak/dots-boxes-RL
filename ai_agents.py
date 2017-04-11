# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 18:06:56 2017

@author: matthew
"""
from naive_players import Player
import numpy as np
from numpy import random
import tensorflow as tf
from dots_and_boxes import DotsAndBoxes

class TDLearner(Player):
    """Monte Carlo Funcion Approximation"""
    
    def __init__(self,name,alpha=0.01,epsilon=0.05,gamma=0.95,lmbda = 1):
        """Initializes the SARSA agent.
        
        Attributes:
            weights: weights used by the linear function approximator. One weight matrix per action.
            epsilon: epsilon value for an epsilon greedy improvement policy (0-1)
            alpha: learning rate [0-1]
            lmbda: lambda for SARSA(lambda) improvement policy - determines strength of eligibility trace [0-1]
            eligibility: eligibility table. One per action.
            gamma: Decay rate of reward
            action: Next action to be taken in the environment
        """
        super().__init__()
        self.name = name
        self.epsilon = epsilon
        self.alpha = alpha
        self.learning = True
        self.lmbda = lmbda
        self.gamma = gamma
        self.save_count = 0
        self.DQN = []
        
        self.sess = None
        # Variables needed for Q function
        self.Q_value = None
        self.input_matrix = None
        self.learning = True
        
        self.last_state = None
        self.last_action = None
        self.saver = None
        self.target_Q = None
        self.optimizer = None
        self.update_model = None

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self, environment):
        self._environment = environment
        if self.sess is None:
            self._build_model()
        
    def act(self):
        """Chooses an action according to the SARSA lambda algorithm
        """
        state = self._environment.state
        feature_vector = self.generate_input_vector(state)
        
        #TD-Update with 0
        if self.last_state is not None and self.learning:
            self.td_update(self.last_state, self.last_action, feature_vector, 0)
        
        # Choose an action
        action = self.choose_action(feature_vector)
        
        self.last_state = feature_vector
        self.last_action = action
        
        self._environment.step(action)
        
        return action


    def generate_input_vector(self, state):
        r, c, d = state.shape
        input_vector = np.reshape(state, [1, r, c, d])
        return input_vector

    def receive_reward(self, reward):
        if self.learning:
            self.td_update(self.last_state, self.last_action, None, reward)
            
    def choose_action(self, feature_vector):
        """Chooses an action based on an epsilon-greedy SARSA policy"""
        if random.random() < self.epsilon and self.learning:
            chosen_action = np.random.choice(self._environment.valid_actions)
        else:
            q_values = self.get_Q_values(feature_vector)
            best_actions = np.where(q_values == q_values[self._environment.valid_actions].max())[0]
            chosen_action = random.choice(best_actions)
       
        return chosen_action

    def get_Q_values(self, feature_vector):
        """Returns the Q_values of a state"""
        q_values = self.sess.run(self.Q_values, feed_dict={self.input_matrix: feature_vector})[0]
        return q_values
       
    def td_update(self, feature_vector, action, next_feature_vector, reward):
        """Updates the Q_function according to the SARSA update algorithm"""
        # If the next state is terminal, we only care about the reward
        if next_feature_vector is None:
            q_next = np.zeros(len(self.environment.action_list))
        else:
            q_next = self.get_Q_values(next_feature_vector)
            
        # Generate a target Q value
        target = self.gamma*q_next
        ## Update only the action we took
        target[action] += reward
        
        # Update Q model according to target
        self.sess.run(self.update_model, feed_dict={self.target_Q: target, self.input_matrix: feature_vector})

    def save_model(self,checkpoint_name=None, global_step=None):
        """Saves a model and returns the name of the checkpoint"""
        self.saver.save(self.sess, checkpoint_name, global_step=global_step)

    def load_model(self,model_dir):
        """Restores a model from checkpoint"""
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

    def save_DQN(self):
        """Saves the DQN in a CSV file"""

    def load_DQN(self,path):
        """Load a DQN from CSV file"""


    def get_variable(self,var_name):
        variable = self.sess.run(var_name)
        return variable

    def list_variables(self):
        return [v.name for v in tf.trainable_variables()]
        
    def _build_model(self):
        """Generates the neural network for the Q Function"""
        tf.reset_default_graph()
        self.sess = tf.Session()
        
        
        row, column, depth = self._environment.state.shape
        self.input_matrix = tf.placeholder(tf.float32, [None, row, column, depth],name='X')
        output_size = len(self._environment.action_list)
        conv1_shape = [3, 3, 4, 12]
        conv2_shape = [2, 2, 12, 36]

        # Set up weights and biases for convolutional layers
        W1 = tf.Variable(tf.truncated_normal(conv1_shape, stddev=0.1),name='W1')
        B1 = tf.Variable(tf.truncated_normal([12], stddev=0.1),name='B1')
        W2 = tf.Variable(tf.truncated_normal(conv2_shape, stddev=0.1),name='W2')
        B2 = tf.Variable(tf.truncated_normal([36], stddev=0.1),name='B2')

        # Helper function for conv layers
        def conv2d(x,W,strides=[1, 1, 1, 1]):
            return tf.nn.conv2d(x, W, strides=strides, padding='SAME')
        
        # Create convolutional layers
        h1 = tf.nn.relu(tf.add(conv2d(self.input_matrix, W1, strides=[1, 1, 1, 1]), B1),name='Conv1')
        h2 = tf.nn.relu(tf.add(conv2d(h1, W2), B2),name='Conv2')

        # Create flattened layer
        h1_shape = h1.get_shape().as_list()[1:]
        flattened = tf.reshape(h1, [-1, np.prod(h1_shape)], name='Flattened')

        # Create FC and output variables
        W3 = tf.Variable(tf.truncated_normal([flattened.get_shape().as_list()[1], 128], stddev=0.1),name='W3')
        B3 = tf.Variable(tf.truncated_normal([128], stddev=0.1),name='B3')
        #W4 = tf.Variable(tf.truncated_normal([200, 200], stddev=0.1), name='W4')
        #B4 = tf.Variable(tf.truncated_normal([200], stddev=0.1), name='B3')
        outputW = tf.Variable(tf.truncated_normal([128, output_size], stddev=0.1),name='outputW')
        outputB = tf.Variable(tf.truncated_normal([output_size], stddev=0.1),name='outputB')

        # Create FC and output layer
        h3 = tf.nn.relu(tf.add(tf.matmul(flattened, W3), B3),name='FC')
        #h4 = tf.nn.relu(tf.add(tf.matmul(h3,W4), B4), name='FC2')
        output = tf.add(tf.matmul(h3, outputW), outputB,name='output')

        # Assign to output
        self.Q_values = output

        # Create update structure
        self.target_Q = tf.placeholder(tf.float32,[output_size],'Target')
        loss = tf.reduce_mean(tf.square(self.target_Q - self.Q_values))
        self.optimizer = tf.train.AdamOptimizer(self.alpha)
        self.update_model = self.optimizer.minimize(loss)
        
        self.saver = tf.train.Saver()
        
        # Initialize all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        



if __name__ == '__main__':
    s = TDLearner()

        
        
        
        
        