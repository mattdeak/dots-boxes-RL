"""
Module allows the user to play a game against the fully trained agent
"""
import argparse
from environment import DotsAndBoxes
from naive_players import *
from ai_agents import DQNLearner
from simulation_utils import model_path, clone, test, switch_players


def main():
    """
    Main Program: Plays against the trained agent on a 5x5 board
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--player-choice', type=int, choices=[1,2], default=1,
                        help="Select which player you'd like to be (1 or 2)")
    args = parser.parse_args()
    manual_play(args.player_choice)


def manual_play(player1=1):
    """
    Allows a user to play a game against the trained DQN agent in the python console
    :param player1:
    """
    ENVIRONMENT_SIZE = 4
    env = DotsAndBoxes(ENVIRONMENT_SIZE)
    player = ManualPlayer('player')
    opponent = DQNLearner('opponent')
    model_dir = model_path(ENVIRONMENT_SIZE) + '-1000000'

    if player1 == 1:
        env.player1 = player
        env.player2 = opponent
    else:
        env.player1 = opponent
        env.player2 = player

    opponent.initialize_network()
    opponent.learning = False

    try:
        # Load model
        opponent.load_model(model_dir)
    except:
        raise ValueError ("Failed to load model from path {}".format(model_dir))

    env.initialize_game()
    winner = ""

    while env.state is not None:
        print(env)
        acting_player = str(env.current_player)
        action = env.current_player.act()
        print("{} built wall {}".format(acting_player, action))

    if env.exit_code == 'loss':
        if env.current_player == env.player1:
            winner = env.player2
        else:
            winner = env.player1
    else:
        if env.score[env.player1] > env.score[env.player2]:
            winner = env.player1
        elif env.score[env.player1] == env.score[env.player2]:
            winner = None
        else:
            winner = env.player2

    print("Winner: {}".format(winner))

if __name__ == '__main__':
    main()