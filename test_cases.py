# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 00:09:07 2017

@author: matthew
"""

from dots_and_boxes import DotsAndBoxes
from naive_players import Player, ManualPlayer, SimplePlayer
from ai_agents import TDLearner
from collections import defaultdict

def box_made_test():
    """Test cases for making boxes"""
    db = DotsAndBoxes(4)
    
    assert not db.made_box([0,0,1])
    db.state[0,0,1] = 1
    
    assert not db.made_box([0,0,0])
    db.state[0,0,0] = 1
    
    assert not db.made_box([1,0,1])
    db.state[1,0,1] = 1
    
    print (db.state)
    
    assert db.made_box([0,1,0]),'Did not make box at first assertion'
    db.state[1,1,1] = 1
    
    db = DotsAndBoxes()
    
def wall_to_states_test():
    
    db = DotsAndBoxes(3)
    
    for i in range((24)):
        print("Wall Number {} can be represented by states {}".format(i,db.convert_to_state(i)))
    
def state_to_wall_test():
    
    db = DotsAndBoxes(3)
    
    for row in range(3):
        for column in range(3):
            for direction in range(4):
                print("State {} can be represented by wall {}".format([row,column,direction],db.convert_to_wall([row,column,direction])))
                
                
def random_game():
    p1 = Player()
    p2 = Player()
    env = DotsAndBoxes(3)
    
    env.player1 = p1
    env.player2 = p2
    
    log = env.play(log=True)
    for entry in log:
        print(entry)
        
def manual_game():
    p1 = ManualPlayer()
    p2 = SimplePlayer()
    env = DotsAndBoxes(3)
    
    env.player1 = p1
    env.player2 = p2
    
    log = env.play(log=True)
    for entry in log:
        print(entry)
        
def random_vs_simple():
    p1 = SimplePlayer()
    p2 = SimplePlayer()
    env = DotsAndBoxes(4)
    
    env.player1 = p1
    env.player2 = p2
    wins = defaultdict(int)
    for i in range(1000):
        log,winner = env.play(log=True)
        wins[winner] += 1
        
    print (wins)

def SARSATest():
    p1 = SimplePlayer()
    p2 = TDLearner('train')
    env = DotsAndBoxes(5)

    env.player1 = p1
    env.player2 = p2
    s,d,l = env.play(True)
    print (p2.list_variables())
    print (p2.get_variable('outputB:0'))
    return d

def SARSA_Q_val_test():
    p1 = SimplePlayer()
    env = DotsAndBoxes(5)

    env.player1 = p1
    env.player2 = p2
    feature_vector = p2.generate_input_vector(env.state)
    print(p2.get_Q_values(feature_vector))
    
if __name__ == '__main__':
    d = SARSATest()
    print(d)
    
    
    
    