import numpy as np
from random import randrange
from scipy import optimize

# TODO: Schwierigkeitsgrad + unterschiedliche Arten der Spiele selektierbar machen


class Game(object):

    def __init__(self, maximum_int=10, lin=5, col=5, mode=0):
        self.__mode = mode
        self.__maximum_int = maximum_int
        self.__minimum_int = maximum_int * -1
        self.__lin = randrange(2, lin)
        self.__col = randrange(2, col)
        self.__matrix = np.zeros((self.__lin, self.__col))
        self.__matrix2 = np.zeros((self.__lin, self.__col))
        self.fill_matrix(self.__mode)

    def get_matrix(self):
        return self.__matrix

    def set_matrix(self, matrix):
        self.__matrix = matrix

    def set_lines(self, lin):
        self.__lin = lin
        self.fill_matrix()

    def set_columns(self, col):
        self.__col = col
        self.fill_matrix()

    def set_mode(self, mode):
        self.__mode = mode
        self.fill_matrix(mode)

    def set_max_int(self, max_int):
        self.__maximum_int = max_int
        self.__minimum_int = max_int * -1

    def fill_matrix(self, mode=0):
        # Zwei Personen Nullsummenspiele (Modus: 0)
        if mode == 0:
            for count_lin in range(self.__lin):
                for count_col in range(self.__col):
                    x = randrange(self.__minimum_int, self.__maximum_int + 1)
                    self.__matrix[count_lin][count_col] = x
                    self.__matrix2[count_lin][count_col] = x * -1

        # Nicht kooperative 2-Personenspiele (Modus: 1)
        # TODO: Gefangenendilemma und Kampf der Geschlechter als Untermodus
        elif mode == 1:
            for count_lin in range(self.__lin):
                for count_col in range(self.__col):
                    x = randrange(self.__minimum_int, self.__maximum_int + 1)
                    y = randrange(self.__minimum_int, self.__maximum_int + 1)
                    self.__matrix[count_lin][count_col] = x
                    self.__matrix2[count_lin][count_col] = y

        # Kooperative 2-Personenspiele (Modus: 2)
        # TODO: Kompensation durch Seitenzahlung als Untermodus
        # TODO: Nur verbindliche Absprachen über Strategie möglich
        elif mode == 1:
            for count_lin in range(self.__lin):
                for count_col in range(self.__col):
                    x = randrange(self.__minimum_int, self.__maximum_int + 1)
                    y = randrange(self.__minimum_int, self.__maximum_int + 1)
                    self.__matrix[count_lin][count_col] = x
                    self.__matrix2[count_lin][count_col] = y

    def get_mode(self):
        return self.__mode

    def get_maximum_int(self):
        return self.__maximum_int

    def get_lines(self):
        return self.__lin

    def get_columns(self):
        return self.__col

    # TODO: Modi einbauen.
    # TODO: kooperative und nicht-kooperativte Spiele
    # TODO: Determinierte Spiele
    # TODO: Indeterminierte Spiele