# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 18:06:56 2017

@author: matthew
"""
from naive_players import Player
import numpy as np
from numpy import random
from dots_and_boxes import DotsAndBoxes
from DQN import DQN_CMM

class DQNLearner(Player):
    """DQN"""    
    def __init__(self, name, alpha=1e-4, epsilon=0.05, gamma=0.6):
        """Initializes the SARSA agent.
        Attributes:
            epsilon: epsilon value for an epsilon greedy improvement policy (0-1)
            alpha: learning rate [0-1]
            gamma: Decay rate of reward
            action: Next action to be taken in the environment
        """
        super().__init__()
        self.name = name
        self.learning = True
        
        # Hyper-parameter initialization
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        # Variables needed for Q function interaction
        self.DQN = None

        # For TD-Update
        self.last_state = None
        self.last_action = None

        #Logging
        self.log_file = None

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self, environment):
        """Sets the environment
        If there is no Q-model, builds one based on the
        environments action space"""
        self._environment = environment

    def act(self):
        """Chooses an action according to the DQN and performs a TD update if possible
        """
        state = self._environment.state
        assert state is not None, 'Invalid State Passed for Agent Name: {}'.format(self.name)
        # Generate a feature vector
        feature_vector = self.generate_input_vector(state)
        
         # Choose an action
        action = self.choose_action(feature_vector)
        
        self.last_state = feature_vector.copy()
        self.last_action = action
        
        self._environment.step(action)
        
        return action
        
    def observe(self, state, reward):
        """
        Observe a state-reward pair
        :param state: Current state of the game (nxnxd numpy array)
        :param reward: Reward associated with state (integer)
        """
        if self.learning:
            have_next_turn = int(self._environment.current_player == self)
    
            if self._environment.state is not None:
                next_feature_vector = self.generate_input_vector(state)
            else:
                next_feature_vector = None
            
            self.update(self.last_state, self.last_action, next_feature_vector, reward, have_next_turn)


    def generate_input_vector(self, state):
        """Generates an input vector based on an environment state."""
        r, c, d = state.shape
        input_vector = np.resize(state, [1, r, c, d])
        
        return input_vector
            
    def choose_action(self, feature_vector):
        """
        :param feature_vector: The current state of the environment (nxnxd numpy array)
        :return: action to take (int)
        """
        if random.random() < self.epsilon and self.learning:
            chosen_action = np.random.choice(self._environment.valid_actions)
        else:
            q_values = self.DQN.predict(feature_vector)[0]
            max_valid_q = q_values[self._environment.valid_actions].max()
            best_actions = np.where(q_values == max_valid_q)[0]
            chosen_action = random.choice([action for action in best_actions if action in self._environment.valid_actions])
        return chosen_action
       
    def update(self, current_state, last_action, next_state, reward, have_next_turn):
        """
        :param current_state: The current state of the environment (nxnxd binary numpy array)
        :param last_action: The last action taken (int)
        :param next_state: The future state of the environment (nxnxd binary numpy array)
        :param reward: The reward associated with next_state (float)
        :param have_next_turn: Whether or not the agent has control on the next state (int)
        """
        # Update the replay table
        self.DQN.record_state((current_state, last_action, next_state, reward, have_next_turn))

        # Train the DQN
        self.DQN.train()

    def initialize_network(self):
        assert self._environment is not None, 'Cannot initialize a network without environments'
        input_shape = self._environment.state.shape
        outputs = len(self._environment.action_list)
        self.DQN = DQN_CMM(input_shape, outputs, alpha=self.alpha, gamma=self.gamma)


    def save_model(self, checkpoint_name, global_step=None):
        """Saves a model and returns the name of the checkpoint"""
        path = self.DQN.save_model(checkpoint_name, global_step=global_step)
        return path

    def load_model(self, model_dir):
        """Restores a model from checkpoint"""
        self.DQN.load_model(model_dir)

if __name__ == '__main__':
    s = DQNLearner('train')
    s2 = DQNLearner('test')
    s2.learning = False
    
    env = DotsAndBoxes()
    env.player1 = s
    env.player2 = s2
    env.play()
        
        
        
        
        