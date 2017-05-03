# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:13:24 2017

@author: deakma
"""
from ai_agents import DQNLearner
from naive_players import Player, SimplePlayer
from dots_and_boxes import DotsAndBoxes
from simulation_utils import *
import os

def self_play_simulation(environment,train_agent,target_agent,n_games,update_step,test_agents=None,test_games=None):
    """A training environment for an agent"""

    if test_agents is not None:
        test_env = clone(environment)

    environment.player1 = train_agent
    environment.player2 = target_agent

    model_dir = model_path(env.size)
    log_file = log_path(env.size)

    if not os.path.exists(log_file) or os.stat(log_file) == 0:
        with open(log_file,'a') as file:
            game_start = 1
            file.write('Game Number,Test Agent,Win Percentage,Draw Percentage,Loss Percentage\n')
    else:
        game_start = int(recent_game(log_file)) + 1

    train_agent.initialize_network()
    target_agent.initialize_network()

    # Load previous model if it exists
    try:
        train_agent.load_model(model_dir)
        target_agent.load_model(model_dir)
        print("Load Succeeded")
    except:
        print("Attempted load and failed")

    print ("Starting at game {}".format(game_start))

    for game_number in range(game_start, n_games + 1):

        # Switch who goes first every other round
        environment.player1 = train_agent
        environment.player2 = target_agent

        if game_number % 2 == 0:
            switch_players(environment)

        environment.play()

        # Write to logs every 1000 games
        if game_number % 1000 == 0 and test_agents:
            print("Game {} Test Results".format(game_number))
            with open(log_file, 'a') as file:
                for agent in test_agents:
                    win_percentage, draw_percentage, loss_percentage = test(test_env, train_agent, agent, test_games)
                    file.write('{},{},{},{},{}\n'.format(game_number, agent, win_percentage,draw_percentage, loss_percentage))
                    print()

        
        # Play games agains the old model
        if game_number % update_step == 0:

            # Give the target agent the most recent model
            print("Saving current model")
            train_agent.save_model(model_dir)

            # Load model into target
            print ("Loading model into target")
            target_agent.load_model(model_dir)

    
if __name__ == '__main__':
    game_size = 4
    train_agent = DQNLearner('train',alpha=5e-6,gamma=0.6)

    target_agent = DQNLearner('target')
    target_agent.learning = False

    # Load the testing agents
    test_agent1 = Player(name='random_player')
    test_agent2 = SimplePlayer(name='moderate_player', level=1)
    test_agent3 = SimplePlayer(name='advanced_player', level=2)
    
    env = DotsAndBoxes(game_size)
    n_games = 1000000
    update_step = 1000
    test_games = 1000
    
    self_play_simulation(env, train_agent, target_agent,
                                    n_games, update_step,
                                    [test_agent1,test_agent2,test_agent3], test_games)
                
        
            
        
            
        