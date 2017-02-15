import numpy as np
from random import randrange
from scipy import optimize

# TODO: Schwierigkeitsgrad + unterschiedliche Arten der Spiele selektierbar machen


class Game(object):

    def __init__(self, maximum_int, lin, col):
        self.maximum_int = maximum_int
        self.minimum_int = maximum_int * -1
        self.lin = randrange(2, lin)
        self.col = randrange(2, col)
        self.matrix = np.zeros((self.lin, self.col))
        self.fill_matrix()

    def get_matrix(self):
        return self.matrix

    def fill_matrix(self):
        for count_lin in range(self.lin):
            for count_col in range(self.col):
                x = randrange(self.minimum_int, self.maximum_int + 1)
                self.matrix[count_lin][count_col] = x

