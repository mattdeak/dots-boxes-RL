# -*- coding: utf-8 -*-
"""
A module containing simulation functions to train an ai agent

@author: Matthew Deakos
"""
from ai_agents import DQNLearner
from naive_players import SimplePlayer
from simulation_utils import *
import os


def train_simulation(environment, train_agent, target_agent, n_games, update_step, test_agents=None, test_games=None):
    """
    Runs a simulation to train an agent. If test agents are provided, tests the agent at an interval
    of 1000 iterations and logs the results.
    :param environment: game environment
    :param train_agent: learning agent
    :param target_agent: opponent agent
    :param n_games: number of training games to run
    :param update_step: number of games played until opponent model is updated
    :param test_agents: array of test agents
    :param test_games: number of games to play against test agents
    :return:
    """

    # Create a test environment
    if test_agents is not None:
        test_env = clone(environment)

    # Set the players to the environment
    environment.player1 = train_agent
    environment.player2 = target_agent

    # Get the directory to save/load models and log information
    model_dir = model_path(environment.size)
    log_file = log_path(environment.size)

    # For Debugging
    print ("Model Directory is: {}".format(model_dir))
    print ("Log file is: {}".format(log_file))

    # Start log file if it doesn't exist, otherwise load from last game
    if not os.path.exists(log_file) or os.stat(log_file) == 0:
        with open(log_file,'a') as file:
            game_start = 1
            file.write('Game Number,Test Agent,Win Percentage,Draw Percentage,Loss Percentage\n')
    else:
        last_line = recent_game(log_file)
        if last_line != "Game Number":
            game_start = int(recent_game(log_file)) + 1
        else:
            game_start = 1

    train_agent.initialize_network()
    target_agent.initialize_network()

    # Load previous model if it exists
    try:
        train_agent.load_model(model_dir + '-' + str(game_start - 1))
        target_agent.load_model(model_dir + '-' + str(game_start - 1))
        print("Load Succeeded")
    except:
        print("Attempted load and failed")

    # Debugging
    print ("Starting at game {}".format(game_start))

    # Begin training games
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
                    file.write('{},{},{},{},{}\n'.format(game_number, agent, win_percentage, draw_percentage, loss_percentage))
                    print()

        
        # Play games agains the old model
        if game_number % update_step == 0:

            # Give the target agent the most recent model
            print("Saving current model")
            path = train_agent.save_model(model_dir, global_step=game_number)

            # Load model into target
            print ("Loading model into target")
            target_agent.load_model(path)
            print ()

    print ("Finished!")


def output_comparison(training_games=10000, test_games=500):
    """
    Runs a comparison against 2 agents that are identical aside from the output activation function.
    :param training_games: Number of games to play
    :param test_games: Number of test games to play
    """
    training_env = DotsAndBoxes(4)
    training_env2 = clone(training_env)
    test_env = clone(training_env)

    tanh_player = DQNLearner('tanh output', alpha=1e-6, gamma=0.6)
    linout_player = DQNLearner('linear output', alpha=1e-6, gamma=0.6)
    training_opponent = Player('training opponent')
    training_opponent2 = Player('training opponent 2')

    training_env.player1 = tanh_player
    training_env.player2 = training_opponent
    tanh_player.initialize_network(output='tanh')

    training_env2.player1 = linout_player
    training_env2.player2 = training_opponent2
    linout_player.initialize_network(output='linear')

    test_random = Player('Random')
    test_moderate = SimplePlayer('Moderate', level=1)
    test_advanced = SimplePlayer('Advanced', level=2)

    log_file = '.{0:s}Analysis{0:s}output_comparison.txt'.format(os.sep)
    with open(log_file, 'w') as file:
        file.write('Learning Agent,Test Agent,Win %, Draw %, Loss %\n')

    for game_number in range(1, training_games+1):

        # Switch who goes first every other round
        training_env.player1 = tanh_player
        training_env.player2 = training_opponent

        training_env2.player1 = linout_player
        training_env2.player2 = training_opponent2

        # Switch starting positions
        if game_number % 2 == 0:
            switch_players(training_env)
            switch_players(training_env2)

        training_env.play()
        training_env2.play()

        if game_number % (training_games/20) == 0:
            print("Running Tests at game {}".format(game_number))
            for test_agent in (test_random, test_moderate, test_advanced):
                for player in (tanh_player, linout_player):
                    print("Testing player: {}".format(player))
                    wins, draws, loss = test(test_env, player, test_agent, test_games)
                    with open(log_file, 'a') as file:
                        file.write('{},{},{},{},{}\n'.format(player, test_agent, wins, draws, loss))
                    print()

    print ("Training Completed!")

if __name__ == '__main__':
    game_size = 4
    train_agent = DQNLearner('train',alpha=1e-6,gamma=0.6)

    target_agent = DQNLearner('target')
    target_agent.learning = False

    # Load the testing agents
    test_agent1 = Player(name='random_player')
    test_agent2 = SimplePlayer(name='moderate_player', level=1)
    test_agent3 = SimplePlayer(name='advanced_player', level=2)
    
    env = DotsAndBoxes(game_size)
    n_games = 1000000
    update_step = 25000
    test_games = 1000
    
    train_simulation(env, train_agent, target_agent,
                                    n_games, update_step,
                                    [test_agent1,test_agent2,test_agent3], test_games)
                
        
            
        
            
        
