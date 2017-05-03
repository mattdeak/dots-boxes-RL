from copy import deepcopy
from dots_and_boxes import DotsAndBoxes
from naive_players import Player
import os

def test(test_env, train, test, test_games):
    """Test and print out results."""
    train.learning = False
    test_env.player1 = train
    test_env.player2 = test
    test_winners = []
    games = []
    states = []

    for test_game in range(test_games):
        # print("Test Game: {}".format(test_game))
        game, winner, game_length, state_log = test_env.play(log=True)
        test_winners.append(winner)
        games.append(game)
        states.append(state_log)
        switch_players(test_env)

    win_percentage = float(test_winners.count(train.name)) / test_games
    draw_percentage = float(test_winners.count('None')) / test_games
    loss_percentage = float(test_winners.count(test.name)) / test_games

    print("Current win percentage over agent {}: {:.2f}%".format(test.name, win_percentage * 100))
    print("Current draw percentage over agent {}: {:.2f}%".format(test.name, draw_percentage * 100))
    print("Current loss percentage over agent {}: {:.2f}%".format(test.name, loss_percentage * 100))

    train.learning = True

    return win_percentage, draw_percentage, loss_percentage


def switch_players(env):
    """Switches player position"""
    p1 = env.player1
    p2 = env.player2
    env.player1 = p2
    env.player2 = p1


def clone(obj):
    """Creates a clone of the environment for testing"""
    return deepcopy(obj)

def switch_test():
    e = DotsAndBoxes(4)
    e.player1 = Player('one')
    e.player2 = Player('two')

    print (e.current_player)
    print (e.player1)
    print (e.other_player)
    print (e.player2)
    switch_players(e)

    print (e.current_player)
    print (e.player1)
    print (e.other_player)
    print (e.player2)

def recent_game(log_file):
    with open(log_file,'r') as file:
        lines = file.readlines()
        number = lines[-1].split(',')[0]
    return number

def model_path(size):
    return './models{0:s}size{1:d}{0:s}'.format(os.sep, size)

def log_path(size):
    return './models{0:s}size{1:d}{0:s}logs.txt'.format(os.sep, size)

if __name__ == '__main__':
    print(model_path(3))
