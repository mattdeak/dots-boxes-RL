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
        
        #Take an action randomly
        action = self.choose_action()
        self.environment.step(action)
        return action
        
    def receive_reward(self,reward):
        """Random player does nothing"""
        
    def choose_action(self):
        """Choose an action randomly in the environment."""
        action = random.choice(self.environment.valid_actions)
        
        return action
        
        
class ManualPlayer(Player):
    """A player that interacts with the environment according to user input"""
        
    def choose_action(self):
        """Choose an action based on user input"""
        print ("Current Game Board:\n{}".format(self.environment))
        action = None
        while action not in self.environment.valid_actions:
            print("valid action: {}".format(self.environment.valid_actions))
            action = int(input("Please choose an action\n>>"))
        
        return action
        
    
    
        
        