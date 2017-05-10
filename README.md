# dots-boxes-RL
Dots and Boxes - A Machine Learning Approach

## Requirements
* Python 3
* Numpy
* Tensorflow 1.1

## Instructions

#### Playing against a trained model
After cloning the repository, simply navigate to the folder and run

`python3 play.py`

This command will bring up a console-based game of dots and boxes where you can play against the trained model on a 5 dot x 5 dot board.

You can also elect to be player 2 by using the optional parameter `player-choice`

`python3 play.py --player-choice 2`

In-game, build walls at selected locations according to the key map shown below:

PIC HERE

## Training a Model

To train a model, simply run the command

`python3 simulation.py`

Currently, this file is hard-coded to halt at one million iterations. Since the current model has already been trained on one-million
iterations, you must either delete the previous model (located in the models/size4 subdirectory) or manual change the iteration limit
in the simulation.py main program if you wish to run a training simulation. This parameter is denoted by `n_games`.




