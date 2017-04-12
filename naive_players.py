# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:18:36 2017

@author: matthew
"""

from numpy import random

class Player:
    """The base class for players integrating with an environment.
    """

    def __init__(self,name=None):
        """Initializer for the base interactive agent.
        
        Attributes:
            environment: The environment that this class is acting on. 
        """
        self._environment = None
        self.name = name

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self,environment):
        self._environment = environment
    
    def __str__(self):
        return self.name
        
        
    def act(self):
        """Takes a random action in the environment."""
        if self._environment == None:
            raise ValueError("Must add an environment in order to act")
        
        #Take an action randomly
        action = self.choose_action()
        self._environment.step(action)
        return action
        
    def observe(self,last_state,action,reward):
        """Random player does nothing"""
        
    def choose_action(self):
        """Choose an action randomly in the environment."""
        action = random.choice(self._environment.valid_actions)
        
        return action
        
        
class ManualPlayer(Player):
    """A player that interacts with the environment according to user input"""
        
    def choose_action(self):
        """Choose an action based on user input"""
        print ("Current Game Board:\n{}".format(self._environment))
        action = None
        while action not in self._environment.valid_actions:
            print("valid action: {}".format(self._environment.valid_actions))
            action = int(input("Please choose an action\n>>"))
        
        return action
        
class SimplePlayer(Player):
    """AI that plays the game based on a simple heuristic.
    
    When the SimplePlayer detects that it can score a point through one of its actions, it will do so.
    It will also try to avoid putting the third wall on a box."""
    
    def choose_action(self):
        fourth_wall_actions = []
        third_wall_actions = []

        for action in self._environment.valid_actions:
            for wall_count in self.analyze_action(action):
                if wall_count == 4:
                    fourth_wall_actions.append(action)
                if wall_count == 3:
                    third_wall_actions.append(action)
        #We want to avoid putting the third wall on any box, because that means the opponent can score          
        safe_actions = [a for a in self._environment.valid_actions if a not in third_wall_actions]
        
        if len(fourth_wall_actions) != 0:
            action = random.choice(fourth_wall_actions)
        elif len(safe_actions) != 0:
            action = random.choice(safe_actions)
        else:
            action = random.choice(self._environment.valid_actions)
            
        return action
            

    def analyze_action(self,action):
        """Returns the count of walls of affected cells IF an action is taken.
        
        Example: If an action will place a wall that completes a box on one cell
        and completes the third wall on another, return [4,3]
        """
        affected_cells = self._environment.convert_to_state(action)
        resulting_wall_count = []
        for cell in affected_cells:
            row,column,side = cell
            current_walls = sum(self._environment.state[row,column])
            resulting_wall_count.append(current_walls + 1)
            
        return resulting_wall_count
        
    
    
        
        