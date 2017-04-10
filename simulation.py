# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:13:24 2017

@author: deakma
"""
from ai_agents import TDLearner
from naive_players import Player
from dots_and_boxes import DotsAndBoxes
from copy import deepcopy
import os

def self_play_simulation(environment,train_agent,target_agent,n_games,update_step,test_agent=None,test_games=None):
    """A training environment for an agent"""
    
    def clone(obj):
        """Creates a clone of the environment for testing"""
        return deepcopy(obj)
        
    if test_agent is not None:
        test_env = clone(environment)
        
    
    def switch_players(environment):
        """Switches player position"""
        if environment.player1 == train_agent:
            environment.player2 = train_agent
            environment.player1 = target_agent
            
    game_logs = []
    test_logs = []
    for game_number in range(n_games):
        
        environment.player1 = train_agent
        environment.player2 = target_agent
        
        #Play games agains the old model
        for games_since_last_update in range(update_step):
            #print("Playing Game: {}".format(games_since_last_update))
            game,winner = environment.play(log=True)
            game_logs.append((game,winner))
            switch_players(environment)
        
        #Test performance against a test agent
        if test_agent is not None:
            
            test_env.player1 = train_agent
            test_env.player2 = test_agent
            
            for test_game in range(test_games):
                #print("Test Game: {}".format(test_game))
                game,winner = test_env.play(log=True)
                test_logs.append((game,winner))
                switch_players(test_env)
        
        #Give the target agent the most recent model
        current_model = train_agent.save_model('{}\\model_at_game_{}'.format(os.getcwd(),game_number))
        target_agent.load_model(current_model)
                
    return game_logs, test_logs
    

    
if __name__ == '__main__':
    game_size = 4
    train_agent = TDLearner('train')
    target_agent = TDLearner('test')
    test_agent = Player()
    env = DotsAndBoxes(game_size)
    n_games = 1000
    update_step = 100
    test_games = 10
    logs, tests = self_play_simulation(env,train_agent,target_agent,n_games,update_step,test_agent,test_games)
                
        
            
        
            
        