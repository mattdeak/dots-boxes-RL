# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:18:36 2017

@author: matthew
"""

from numpy import random

class Player:
    """The base class for players integrating with an environment.
    """

    def __init__(self):
        """Initializer for the base interactive agent.
        
        Attributes:
            environment: The environment that this class is acting on. 
        """
        self.environment = None
        self.name = None
        
    def __str__(self):
        return self.name
        
        
    def act(self):
        """Takes a random action in the environment."""
        if self.environment == None:
            raise ValueError("Must add an environment in order to act")
        
        state = self.environment.state
        #Take an action randomly
        action = self.choose_action(state)
        self.environment.step(action)
        return action
        
    def receive_reward(self,reward):
        """Random player does nothing"""
        
    def choose_action(self,state):
        """Choose an action randomly in the environment."""
        action = random.choice(self.environment.valid_actions)
        
        return action