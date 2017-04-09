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
        self.current_player = 1
        self.score = {'player1':0,'player2':0}
        self.action_list = range(size * (size + 1))
        self.turn = 1 #Which player's turn is it
        self.manual = 0 #No manual player
        self.reward_dictionary = {'win':1,'loss':-1,'draw':0}
        self.rewards = {1:0,2:0}
        self.SIDES = {'N':0,'S':1,'E':2,'W':3}
        
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
        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1
          
    
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
        self.build_wall(action)
        
        #Determine the score of the action
        score = self.action_score(action)
        
        if self.action_score(action) > 0:
            self.score[self.current_player - 1] += score
        else:
            self.switch_turn()
        
    
    def action_score(self,action):
        """Determine whether the action made a box and returns the score"""
        states = self.convert_to_state(action)
        score = 0
        
        for state_index in states:
            cell_row,cell_column = state_index[:2]
            if np.sum(self.state[cell_row,cell_column]) == 4:
                score += 1
        
        return score
        
        
    def play(self):
        """Plays a game"""
        
    
    def reward_payout(self):
        self._player1.receive_reward()
        
    def build_wall(self,action):
        """Builds a wall in the game state"""
        states = self.convert_to_state(action)
        for state_index in states:
            self.state[state_index] = 1
        
    def convert_to_wall(self,state):
        """Converts a specific game state array to the wall it represents"""
        row,column,side = state
        if side == self.SIDES['N']:
            return row * self.size + column
        elif side == self.SIDES['S']:
            return (row + 1) * self.size + column
        elif side == self.SIDES['E']:
            return (self.size * (self.size + 1)) + row * self.size + column
        elif side == self.SIDES['W']:
            return (self.size * (self.size + 1)) + row * self.size + (column + 1)
        else:
            raise ValueError("Can't convert state to wall. 3rd Dimension must be range [0-3]. Value given: {}".format(side))
            
    
    def convert_to_state(self,wall_number):
        """Converts a wall number to the specific game states that it represents"""
        #If this is true, the wall is on the N-S
        cells = []
        if wall_number < ((self.size) * (self.size + 1)):
            cell_column = wall_number % self.size
            cell_row = wall_number // self.size
            if cell_row != 0:
                cells.append([cell_row - 1, cell_column, self.SIDES['S']])
            if cell_row != self.size:
                cells.append([cell_row, cell_column, self.SIDES['N']])
        #Otherwise the wall is E-W        
        else:
            cell_row = (wall_number-(self.size * (self.size + 1)))//(self.size + 1)
            cell_column = wall_number % (self.size + 1)
            if cell_column != 0:
                cells.append([cell_row, cell_column - 1, self.SIDES['E']])
                
            if cell_column != self.size:
                cells.append([cell_row, cell_column, self.SIDES['W']])
                
        return cells
                

    def is_valid_action(self,action):
        """Ensures that an action is valid"""
        states = self.convert_to_state(action)
        for state_index in states:
            if self.state[state_index] == 1:
                return False
        return True
        
    def print_state(self):
        """Provides a console output of the current state"""
        walls = []
        for row in range(self.size):
            for column in range(self.size):
                for side in self.state[row,column]:
                    if side == 1:
                        walls.append(self.convert_to_wall([row,column,side]))
                        
        print(np.unique(walls))
        
            
            
            
                
if __name__ == '__main__':
    db = DotsAndBoxes(3)
    db.print_state()
    print('----------------------')
    db.state[1,1,1] = 1
    db.state[1,1,0] = 1
    db.print_state()
        
        
        