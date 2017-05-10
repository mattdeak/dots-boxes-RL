"""
This module provides some utility functions for the simulation module
"""

from copy import deepcopy
from environment import DotsAndBoxes
from naive_players import Player
import os

def test(test_env, train, test, test_games, advanced_logs=False):
    """
    Tests an environment against a test agent
    :param test_env: game environment
    :param train: learning agent
    :param test: testing agent
    :param test_games: number of test games to play (integer)
    :return: evaluation metrics
    """

    # If the agent is set to learn, make sure it's switched back on before the function completes
    restart_learn = train.learning == True

    train.learning = False
    test_env.player1 = train
    test_env.player2 = test
    test_winners = []
    games = []
    states = []
    scores = []

    for test_game in range(test_games):
        # print("Test Game: {}".format(test_game))
        game, winner, game_length, state_log, final_score = test_env.play(log=True)
        test_winners.append(winner)
        games.append(game)
        states.append(state_log)
        scores.append(final_score)
        switch_players(test_env)

    win_percentage = float(test_winners.count(train.name)) / test_games
    draw_percentage = float(test_winners.count('None')) / test_games
    loss_percentage = float(test_winners.count(test.name)) / test_games

    print("Current win percentage over agent {}: {:.2f}%".format(test.name, win_percentage * 100))
    print("Current draw percentage over agent {}: {:.2f}%".format(test.name, draw_percentage * 100))
    print("Current loss percentage over agent {}: {:.2f}%".format(test.name, loss_percentage * 100))

    if restart_learn:
        train.learning = True

    if not advanced_logs:
        return win_percentage, draw_percentage, loss_percentage
    else:
        return win_percentage, draw_percentage, loss_percentage, games, states, test_winners, scores


def switch_players(env):
    """
    Switches player position in environment
    :param env: A game environment
    """
    p1 = env.player1
    p2 = env.player2
    env.player1 = p2
    env.player2 = p1

def clone(obj):
    """Creates a clone of the environment for testing"""
    return deepcopy(obj)


def recent_game(log_file):
    """
    Gets the most recent game as indicated in the log file
    :param log_file: file path
    :return: game number
    """
    with open(log_file,'r') as file:
        lines = file.readlines()
        number = lines[-1].split(',')[0]
    return number

def model_path(size):
    """
    Gets the path to the model directory
    :param size: size of the environment
    :return: path to model directory
    """
    return '.{0:s}models{0:s}size{1:d}{0:s}'.format(os.sep, size)

def log_path(size):
    """
    Gets the path to the log file
    :param size: size of the environment
    :return: log path
    """
    return '.{0:s}models{0:s}size{1:d}{0:s}logs.txt'.format(os.sep, size)


