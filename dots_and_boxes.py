# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:52:43 2017

@author: matthew
"""

import numpy as np
from collections import defaultdict
from numpy import random
from time import sleep

class DotsAndBoxes():
    """The main environment for a dots and boxes game"""
    

    
    def __init__(self,size=2):
        """Initializer"""
        self.size = size
        #State is dot-to-dot cells with 4 channels (up,down,left,right)
        self.state = np.zeros([size,size,4])
        self._player1 = None
        self._player2 = None
        self.current_player = self._player1
        self.score = defaultdict(int)
        self.action_list = range(size * (size + 1) * 2)
        self.valid_actions = None
        self.manual = 0 #No manual player
        self.reward_dictionary = {'win':1,'loss':-1,'draw':0}
        self.captured_cells = defaultdict(list)
        self.exit_code = None
        self.SIDES = {'N':0,'S':1,'E':2,'W':3}
        
    @property    
    def player1(self):
        """Player 1 property getter"""
        return self._player1
        
    @property
    def player2(self):
        """Player 2 property getter"""
        return self._player2 
        
    @player1.setter
    def player1(self,agent):
        """Sets the player and adds current game instance as the agent environment"""
        agent.environment = self
        self._player1 = agent
        if self._player1.name is None:
            self._player1.name = "Player 1"
        self.current_player = self._player1
      
    @player2.setter
    def player2(self,agent):
        """Sets the player and adds current game instance as the agent environment"""
        agent.environment = self
        self._player2 = agent
        if self._player2.name is None:
            self._player2.name = "Player 2"
        
    def switch_turn(self):
        """Switches the turn"""
        if self.current_player == self._player1:
            self.current_player = self._player2
        else:
            self.current_player = self._player1
            
    def payout(self):
        """Updates the player rewards"""
        if self.score[self._player1] < self.score[self._player2]:
            self._player1.receive_reward(self.reward_dictionary['loss'])
            self._player2.receive_reward(self.reward_dictionary['win'])
        elif self.score[self._player1] == self.score[self._player2]:
            self._player1.receive_reward(self.reward_dictionary['draw'])
            self._player2.receive_reward(self.reward_dictionary['draw'])
        else:
            self._player1.receive_reward(self.reward_dictionary['win'])
            self._player2.receive_reward(self.reward_dictionary['loss'])
            
    def end_game(self):
        """Ends the Game"""
        self.state = None
        if self.exit_code is None:
            self.payout()
        else:
            self.current_player.receive_reward(self.reward_dictionary[self.exit_code])
        
    
    
    def step(self,action):
        """Takes an action and changes the game state."""
        #If an action is invalid and the environment is set to quit
        #on an invalid action, set the reward to loss and return
        #the terminal state. Otherwise, simply return the state without
        #switching turns, to give the player another chance.
        if (not self.is_valid_action(action)) and self.current_player.learning:
                raise ValueError("Should not be here anymore!")
                self.exit_code = 'loss'
                self.end_game()
        else:
            #If the agent is not learning, just choose a random action
            if not self.is_valid_action(action):
                action = random.choice(self.valid_actions)
            #Add a wall where the action dictates
            self.build_wall(action)
            
            #Determine the score of the action
            scored = self.score_action(action)
            #Remove the action from the list of valid actions
            self.valid_actions.remove(action)            
            if self.valid_actions == []:
                self.end_game()
            else:
                if not scored:
                    self.switch_turn()
        
    
    def score_action(self,action):
        """Scores the action and returns whether"""
        states = self.convert_to_state(action)
        scored = False
        
        for state_index in states:
            cell_row,cell_column = state_index[:2]
            if np.sum(self.state[cell_row,cell_column]) == 4:
                self.captured_cells[self.current_player].append([cell_row,cell_column])
                self.score[self.current_player] += 1
                scored = True
                
        return scored

    def play(self, log=False, pause=None):
        """Plays a game"""
        self._initialize_game()
        game_log = []
        state_log = []
        game_length = 0
        winner = ""
        
        while self.state is not None:
            if pause is not None:
                sleep(pause)
            if log:
                game_log.append("\nState:\n{}\n".format(self))
                acting_player = str(self.current_player)
                state_log.append(self.state)
                action = self.current_player.act()
                game_log.append("{} built wall {}".format(acting_player, action))
                game_log.append("Current Player 1 Score: {}".format(self.score[self._player1]))
                game_log.append("Current Player 2 Score: {}".format(self.score[self._player2]))

                game_length += 1
            else:
                self.current_player.act()
                
        if log:
            if self.exit_code == 'loss':
                if self.current_player == self._player1:
                    winner = self._player2
                else:
                    winner = self._player1
            else:
                if self.score[self._player1] > self.score[self._player2]:
                    winner = self._player1
                elif self.score[self._player1] == self.score[self._player2]:
                    winner = None
                else:
                    winner = self._player2

            game_log.append("Winner: {}".format(winner))
                
        return game_log, str(winner), game_length, state_log

    def _initialize_game(self):
        self.valid_actions = [i for i in range(self.size * (self.size + 1) * 2)]
        self.score = defaultdict(int)
        self.exit_code = None
        self.state = np.zeros([self.size,self.size,4])
        
    
    def reward_payout(self):
        self._player1.receive_reward()
        
    def build_wall(self,action):
        """Builds a wall in the game state"""
        states = self.convert_to_state(action)
        for state_index in states:
            row,column,side = state_index
            self.state[row,column,side] = 1
        
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
        cells = []
        #If this is true, the wall is on the N-S
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
            r,c,s = state_index
            if self.state[r,c,s] == 1:
                return False
        return True
        
    def __str__(self):
        """Provides a console output of the current state"""
        string = ""
        final_row = []
        for row in range(self.size):
            column_walls = []
            
            for column in range(self.size):
                string += ("{:<1}".format("."))
                n,s,e,w = self.state[row,column]
                n_string = ""                
                if n == 1:
                    n_string = "----"
                else:
                    n_string = " "
                string += "{:<4}".format(n_string)
                
                if w == 1:
                    column_walls.append("{:<1}".format("|"))
                else:
                    column_walls.append("{:<1}".format(" "))
                    
                if e == 1 and column == self.size - 1:
                    column_walls.append("{:<1}".format("|"))
                    
                if column == self.size - 1:
                    string += "{:<1}".format(".")
                    
                if s == 1 and row == self.size - 1:
                    final_row.append("{:<4}".format("____"))
                elif s == 0 and row == self.size - 1:
                    final_row.append("{:<4}".format(" "))
                    
                    
            string += "\n"      
            for wall in column_walls:
                string += wall
                string += "{:<4}".format(" ")
            string += "\n"
            
        for wall in final_row:
            string += ("{:<1}".format("."))
            string += wall
            
        string += ".\n"
        
        string += "\nPlayer 1 Score is {}\nPlayer 2 Score is {}\n".format(self.score[self._player1],self.score[self._player2])
                            
        return string

    def print_state(self,state):
        """Provides a console output of the current state"""
        string = ""
        final_row = []
        for row in range(self.size):
            column_walls = []

            for column in range(self.size):
                string += ("{:<1}".format("."))
                n, s, e, w = state[row, column]
                n_string = ""
                if n == 1:
                    n_string = "----"
                else:
                    n_string = " "
                string += "{:<4}".format(n_string)

                if w == 1:
                    column_walls.append("{:<1}".format("|"))
                else:
                    column_walls.append("{:<1}".format(" "))

                if e == 1 and column == self.size - 1:
                    column_walls.append("{:<1}".format("|"))

                if column == self.size - 1:
                    string += "{:<1}".format(".")

                if s == 1 and row == self.size - 1:
                    final_row.append("{:<4}".format("____"))
                elif s == 0 and row == self.size - 1:
                    final_row.append("{:<4}".format(" "))

            string += "\n"
            for wall in column_walls:
                string += wall
                string += "{:<4}".format(" ")
            string += "\n"

        for wall in final_row:
            string += ("{:<1}".format("."))
            string += wall

        string += ".\n"

        string += "\nPlayer 1 Score is {}\nPlayer 2 Score is {}\n".format(self.score[self._player1],
                                                                          self.score[self._player2])

        return string
                    
                        
        
        
            
            
            
                
if __name__ == '__main__':
    db = DotsAndBoxes(3)
    print(db)
    print('----------------------')
    db.build_wall(2)
    db.build_wall(14)
    db.build_wall(15)
    db.build_wall(6)
    db.build_wall(20)
    db.build_wall(9)
    print(db)
        
        
        