import numpy as np
from random import randrange

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

    def get_matrix2(self):
        return self.__matrix2

    def set_matrix(self, matrix):
        self.__matrix = matrix

    def set_matrix2(self, matrix):
        self.__matrix2 = matrix

    def set_lines(self, lin):
        self.__lin = lin
        self.fill_matrix(self.__mode)

    def set_columns(self, col):
        self.__col = col
        self.fill_matrix(self.__mode)

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
            self.__matrix2 = self.__matrix * -1

        # Nicht kooperative 2-Personenspiele (Modus: 1)
        elif mode == 1:
            self.__matrix = np.zeros((self.__lin, self.__col, 2))
            for count_lin in range(self.__lin):
                for count_col in range(self.__col):
                    x = randrange(self.__minimum_int, self.__maximum_int + 1)
                    y = randrange(self.__minimum_int, self.__maximum_int + 1)
                    self.__matrix[count_lin][count_col] = x
                    self.__matrix2[count_lin][count_col] = y

        # Gefangenendilemma
        # 2 Strategien pro Person: Cooperate, Defect
        # Auszahlung Cooperate x Cooperate = (b, b)
        # Auszahlung Cooperate x Defect = (d, a)
        # Auszahlung Defect x Cooperate = (a, d)
        # Auszahlung Defect x Defect = (c, c)
        # a > b > c > d
        # 2 * b > a + d
        elif mode == 11:
            self.__matrix = np.zeros((2, 2))
            self.__matrix2 = np.zeros((2, 2))
            a = randrange(0, 3 + self.__maximum_int + 1)
            b = randrange(a - self.__maximum_int, a)
            c = randrange(b - self.__maximum_int, 2*b - a + 1)
            d = randrange(c - self.__maximum_int, 2*b - a)
            self.__matrix = np.asarray([[b, d],
                                        [a, c]])
            self.__matrix2 = self.__matrix.transpose()


        # Kampf der Geschlechter
        # 2 Strategien pro Person: Alternative 1, Alternative 2
        # Auszahlung Alt 1 x Alt 1 = (a, b)
        # Auszahlung Alt 1 x Alt 2 = (c, c)
        # Auszahlung Alt 2 x Alt 1 = (c, c)
        # Auszahlung Alt 2 x Alt 2 = (b, a)
        # a > b > c
        elif mode == 12:
            self.__matrix = np.zeros((2, 2))
            self.__matrix2 = np.zeros((2, 2))
            a = randrange(0, 3 + self.__maximum_int + 1)
            b = randrange(a - self.__maximum_int, a)
            c = randrange(b - self.__maximum_int, b)
            self.__matrix = np.asarray([[a, c],
                                        [c, b]])
            self.__matrix2 = np.rot90(self.__matrix, 2)


        # Kooperative 2-Personenspiele (Modus: 2)
        # TODO: Kompensation durch Seitenzahlung als Untermodus
        # TODO: Nur verbindliche Absprachen über Strategie möglich
        elif mode == 2:
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

    def get_minimum_int(self):
        return self.__minimum_int

    def get_lines(self):
        return self.__lin

    def get_columns(self):
        return self.__col

    # TODO: Modi einbauen.
    # TODO: kooperative und nicht-kooperativte Spiele
    # TODO: Determinierte Spiele
    # TODO: Indeterminierte Spiele