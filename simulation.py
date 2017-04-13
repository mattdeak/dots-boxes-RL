# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:13:24 2017

@author: deakma
"""
from ai_agents import TDLearner
from naive_players import Player, SimplePlayer
from dots_and_boxes import DotsAndBoxes
from copy import deepcopy
import numpy as np
import os

def self_play_simulation(environment,train_agent,target_agent,n_games,update_step,test_agents=None,test_games=None):
    """A training environment for an agent"""
    
    def clone(obj):
        """Creates a clone of the environment for testing"""
        return deepcopy(obj)

    def test(test_env,train,test,test_games):
        """Test and print out results."""
        train.learning = False
        test_env.player1 = train
        test_env.player2 = test
        test_winners = []
        game_lengths = []
        games = []
        states = []

        for test_game in range(test_games):
            # print("Test Game: {}".format(test_game))
            game, winner, game_length, state_log = test_env.play(log=True)
            test_logs.append((game, winner))
            test_winners.append(winner)
            game_lengths.append(game_length)
            games.append(game)
            #switch_players(test_env)
            states.append(state_log)

        win_percentage = float(test_winners.count(train.name)) / test_games
        draw_percentage = float(test_winners.count('None')) / test_games
        loss_percentage = float(test_winners.count(test.name)) / test_games
        avg_game_length = np.average(game_lengths)

        print("Current win percentage over agent {}: {:.2f}%".format(test.name,win_percentage * 100))
        print("Current draw percentage over agent {}: {:.2f}%".format(test.name,draw_percentage * 100))
        print("Current loss percentage over agent {}: {:.2f}%".format(test.name,loss_percentage * 100))
        print("Average Game Length: {}".format(avg_game_length))

        train.learning = True

        # print("Last Game: ")
        # for game in games[-1]:
        #     print(game)

    if test_agents is not None:
        test_env = clone(environment)
        
    
    def switch_players(env):
        """Switches player position"""
        p1 = env.player1
        p2 = env.player2
        env.player1 = p2
        env.player2 = p1
            
    game_logs = []
    test_logs = []

    environment.player1 = train_agent
    environment.player2 = target_agent

    try:
        train_agent.load_model('C:\\users\\deakma\\Dots-Boxes-RL\\models\\')
        target_agent.load_model('C:\\users\\deakma\\Dots-Boxes-RL\\models\\')
        print("Load Succeeded")
    except:
        print("Attempted load and failed")

    for game_number in range(1,n_games+1):

        environment.player1 = train_agent
        environment.player2 = target_agent

        game, winner, g, state_log = environment.play(log=True)
        game_logs.append((game, winner))
        #switch_players(environment)
        
        #Play games agains the old model
        if game_number % update_step == 0 and test_agents:

            for agent in test_agents:
                test(test_env,train_agent,agent,test_games)
                print()
                # print("Last Game: ")
                # for game_state in states[-1]:
                #     state_string = env.print_state(game_state)
                #     print("\n",state_string)
                #     input_v = train_agent.generate_input_vector(game_state)
                #     print("Q_Values")
                #     q_vals = train_agent.get_Q_values(input_v)
                #     for i in range(q_vals.size):
                #         print("Action {}, {}".format(i,q_vals[i]),end=" ")
                #         if i % 7 == 0:
                #             print()

            #Give the target agent the most recent model
            print("Saving current model")
            train_agent.save_model('C:\\users\\deakma\\Dots-Boxes-RL\\models\\'.format(env.size))

            print("Loading model into target")
            target_agent.load_model('C:\\users\\deakma\\Dots-Boxes-RL\\models\\'.format(env.size))
            
                
    return game_logs, test_logs
    
if __name__ == '__main__':
    game_size = 3
    train_agent = TDLearner('train',alpha=1e-6,gamma=0.6)

    target_agent = TDLearner('target')
    target_agent.learning = False


    test_agent1 = Player('test')
    test_agent2 = SimplePlayer('test_2')
    
    env = DotsAndBoxes(game_size)
    n_games = 500000
    update_step = 1000
    test_games = 100
    
    logs, tests = self_play_simulation(env, train_agent, target_agent,
                                       n_games, update_step,
                                       [test_agent1,test_agent2], test_games)
                
        
            
        
            
        