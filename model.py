
from dots_and_boxes import DotsAndBoxes

class GameModel():

    def __init__(self, size):
        """Initializer for the model"""
        self.env = DotsAndBoxes(size)
        self._player1 = None
        self._player2 = None

    def play(self, pause=0.5):
        """Start a game in the model."""
        self.env.play(pause=pause)

    @property
    def player1(self):
        return self._player1

    @property
    def player2(self):
        return self._player2

    @player1.setter
    def player1(self, agent):
        agent.environment = self.env
        self._player1 = agent

    @player2.setter
    def player2(self,agent):
        agent.environment = self.env
        self._player2 = agent

