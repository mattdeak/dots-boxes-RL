# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:52:43 2017

@author: matthew
"""

import numpy as np
from enum import Enum

class DotsAndBoxes():
    """The main environment for a dots and boxes game"""
    
    DOWN = 0
    RIGHT = 1
    
    def __init__(self,size=2):
        """Initializer"""
        self.size = size
        #State is dot-to-dot cells with 4 channels (up,down,left,right)
        self.state = np.zeros([size,size,4])
        self._player1 = None
        self._player2 = None
        self.player_turn = 2   
        self.score = {'player1':0,'player2':0}
        self.action_list = []
        self.turn = 1 #Which player's turn is it
        self.manual = 0 #No manual player
        self.reward_dictionary = {'win':1,'loss':-1,'draw':0}
        self.rewards = {1:0,2:0}
        
        
    class Sides(Enum):
        N = 0
        NORTH = 0
        S = 1
        SOUTH = 1
        E = 2
        EAST = 2
        W = 3
        WEST = 3
    
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
        
    def switch_turn(self):
        """Switches the turn"""
        if self.turn == 1:
            return 2
        return 1
          
    
    def step(self,action):
        """Takes an action and changes the game state."""
        reward = 0
        
        #If the player is trying to act on the terminal state
        #It means they have lost
        if self.state is None:
            return self.rewards['loss']
        
        #If an action is invalid and the environment is set to quit
        #on an invalid action, set the reward to loss and return
        #the terminal state. Otherwise, simply return the state without
        #switching turns, to give the player another chance.
        if not self.is_valid_action:
            if self.manual != self.player_turn:
                reward = self.rewards['loss']
                self.state = None
            return self.state,reward
        
        #Add a wall where the action dictates
        self.state[action] = 1
        action_score = self.made_box(action)
        if self.made_box(action) > 0:
            self.score[self.player_turn - 1] += action_score
        else:
            self.switch_turn()
        
    
    def made_box(self,action):
        """Determine whether the action made a box"""
        
    def play(self):
        """Plays a game"""
        pass
    
    def reward_payout(self):
        self._player1.receive_reward()
    
    def convert_to_wall(self,state):
        """Converts a specific game state array to the wall it represents"""
        row,column,side = state
        if side == self.Sides.N:
            return row * self.size + column
        elif side == self.Sides.S:
            return (row + 1) * self.size + column
        elif side == self.Sides.E:
            return (self.size * (self.size + 1)) + row * self.size + column
        elif side == self.Sides.W:
            return (self.size * (self.size + 1)) + row * self.size + (column + 1)
        else:
            raise ValueError("Can't convert state to wall. 3rd Dimension must be range [0-3]")
            
    
    def convert_to_state(self,wall_number):
        """Converts a wall number to the specific game states that it represents"""
        #If this is true, the wall is on the N-S
        cells = []
        if wall_number < ((self.size) * (self.size + 1)):
            cell_column = wall_number % self.size
            cell_row = wall_number // self.size
            if cell_row != 0:
                cells.append([cell_row - 1, cell_column, self.Sides.S])
            if cell_row != self.size:
                cells.append([cell_row, cell_column, self.Sides.N])
        #Otherwise the wall is E-W        
        else:
            cell_row = (wall_number-(self.size * (self.size + 1)))//(self.size + 1)
            cell_column = wall_number % (self.size + 1)
            if cell_column != 0:
                cells.append([cell_row, cell_column - 1, self.Sides.E])
                
            if cell_column != self.size:
                cells.append([cell_row, cell_column, self.Sides,W])
                
        return cells
                
     
    
    def is_valid_action(self,action):
        
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
        
        
        