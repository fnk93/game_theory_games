from GameTheory_Game.Game import Game
import numpy as np


class ConstantSumGame(Game):
    def __init__(self, c, maximum_int=10, minimum_int=-10, lin=np.random.randint(2, 5), col=np.random.randint(2, 5)):
        self.__c = c
        Game.__init__(self, maximum_int, minimum_int, lin, col)
        self.fill_matrix()

    def fill_matrix(self):
        self.matrix = np.random.randint(self.minint, self.maxint + 1, size=(self.lines, self.cols))
        self.matrix2 = self.c - self.matrix

    def getc(self):
        return self.__c

    def setc(self, c):
        self.__c = c
        self.fill_matrix()

    c = property(getc, setc, None, "Konstante c für Konstantsummenspiel")
