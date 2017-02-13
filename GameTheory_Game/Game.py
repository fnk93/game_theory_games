import numpy as np
from random import randrange
from scipy import optimize

class Game:

    def __init__(self, maximum_int, lin, col):
        self.__maximum_int = maximum_int
        self.__minimum_int = maximum_int * -1
        self.__lin = randrange(2, lin)
        self.__col = randrange(2, col)
        self.__matrix = np.zeros((self.__lin, self.__col))
        self.fillMatrix()
        self.__determined = False
        self.__determinedIntervall = []
        self.__top_value2 = 0
        self.__top_value1 = 0
        self.isDetermined()
        self.__maximin_strategies1 = []
        self.__maximin_strategies2 = []
        self.solveStrategies()

    def getMatrix(self):
        return self.__matrix

    def fillMatrix(self):
        for count_lin in range(self.__lin):
            for count_col in range(self.__col):
                x = randrange(self.__minimum_int, self.__maximum_int + 1)
                self.__matrix[count_lin][count_col] = x

    def isDetermined(self):
        max_player2 = []
        for count in range(self.__matrix.transpose().shape[0]):
            max_player2.append(max(self.__matrix.transpose()[count]))
        self.__top_value2 = min(max_player2)

        max_player1 = []
        for count in range(self.__matrix.shape[0]):
            max_player1.append(min(self.__matrix[count]))
        self.__top_value1 = max(max_player1)

        if self.__top_value1 == self.__top_value2:
            self.__determined = True
        else:
            self.__determined = False
        self.__determinedIntervall.append(self.__top_value1)
        self.__determinedIntervall.append(self.__top_value2)

        return self.__determined

    def solveStrategies(self):
        for count in range(self.__matrix.shape[0]):
            if min(self.__matrix[count]) == self.__top_value1:
                self.__maximin_strategies1.append(count+1)

        for count in range(self.__matrix.transpose().shape[0]):
            if max(self.__matrix.transpose()[count]) == self.__top_value2:
                self.__maximin_strategies2.append(count+1)

#    def solveMatrix(self):
