from GameTheory_Game.Game import Game
import numpy as np


class PrisonersDilemma(Game):
    def __init__(self, maximum_int=20, minimum_int=-20, lin=2, col=2):
        Game.__init__(self, maximum_int, minimum_int, lin, col)
        self.fill_matrix()

    def fill_matrix(self):
        self.matrix = np.zeros((2, 2))
        self.matrix2 = np.zeros((2, 2))
        a = np.random.randint(0, 3 + self.maxint + 1)
        b = np.random.randint(a - self.maxint, a)
        c = np.random.randint(b - self.maxint, 2 * b - a + 1)
        d = np.random.randint(c - self.maxint, 2 * b - a)
        self.matrix = np.asarray([[b, d],
                                    [a, c]])
        self.matrix2 = self.matrix.transpose()
