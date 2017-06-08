from GameTheory_Game.Game import Game
import numpy as np


class ConstantSumGame(Game):
    def __init__(self, c, maximum_int=10, minimum_int=-10, lin=np.random.randint(2, 5), col=np.random.randint(2, 5)):
        self.__c = c
        Game.__init__(self, maximum_int, minimum_int, lin, col)
        self.fill_matrix()

    def fill_matrix(self):
        for count_lin in range(self.lines):
            for count_col in range(self.cols):
                x = np.random.randint(self.minint, self.maxint + 1)
                self.matrix[count_lin][count_col] = x
        self.matrix2 = self.c - np.asarray(self.matrix)

    def getc(self):
        return self.__c

    def setc(self, c):
        self.__c = c

    def delc(self):
        del self.__c

    c = property(getc, setc, delc, "Konstante c für Konstantsummenspiel")
