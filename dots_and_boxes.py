# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:52:43 2017

@author: matthew
"""

import numpy as np

class DotsAndBoxes():
    """The main environment for a dots and boxes game"""
    
    def __init__(self,size=2):
        """Initializer"""
        self.size = size
        self.state = np.zeros([size,size,2])
        self._player1 = None
        self._player2 = None
        self.player_turn = 2        
        self.action_list = []
        self.turn = 1 #Which player's turn is it
        self.manual = 0 #No manual player
        self.rewards = {'win':1,'loss':-1,'draw':0}
        self._initialize_action_list()
    
    @property    
    def player1(self):
        """Player 1 property getter"""
        return self._player
        
    @property
    def player2(self):
        """Player 2 property getter"""
        return self._player2 
        
    @player1.setter
    def player1(self,agent):
        """Sets the player and adds current game instance as the agent environment"""
        agent.environment = self
        self._player = agent    
      
    @player2.setter
    def player2(self,agent):
        """Sets the player and adds current game instance as the agent environment"""
        agent.environment = self
        self._player2 = agent
        
    def _initialize_action_list(self):
        """initialize valid actions"""
        for i in range(self.size):
            for j in range(self.size):
                if i != self.size - 1:
                    self.action_list.append([i,j,0])
                
                if j != self.size - 1:
                    self.action_list.append([i,j,1])
    
    def switch_turns(self):
        """Switches the turn"""
        if self.turn == 1:
            return 2
        return 1
          
    
    def step(self,action):
        """Takes an action and returns the next game state."""
        reward = 0
        
        #If an action is invalid and the environment is set to quit
        #on an invalid action, set the reward to loss and return
        #the terminal state. Otherwise, simply return the state without
        #switching turns, to give the player another chance.
        if not self.is_valid_action:
            if self.manual != self.player_turn:
                reward = self.rewards['loss']
                self.state = None
            return self.state,reward
    
    
   
    def play(self):
        """Plays a game"""
        pass
    
    
    def is_valid_action(self,action):
        if self.state[action[0],action[1],action[2]] == 0:
            return True
        return False
        
    def print_state(self):
        """Provides a console output of the current state"""
        for row in range(self.size):
            column_bars = []
            for column in range(self.size):
                print(".",end="")
                if self.state[row,column,1] == 1:
                    print("-",end="")
                else:
                    print(" ",end="")
                
                if self.state[row,column,0] == 0:
                    column_bars.append(" ")
                else:
                    column_bars.append("|")
                
            print()
            for c in column_bars:
                print(c,end="")
            print()
                
if __name__ == '__main__':
    db = DotsAndBoxes(3)
    db.print_state()
    print('----------------------')
    db.state[1,1,1] = 1
    db.state[1,1,0] = 1
    db.print_state()
        
        
        