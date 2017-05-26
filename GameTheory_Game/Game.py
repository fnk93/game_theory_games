import numpy as np
from random import randrange
from GameTheory_Game.Solving_Methods import get_calculations_latex
from sympy import Symbol, refine, Q, var
from GameTheory_Game.Solving_Methods import solve_using_nggw
from sympy.solvers import solve
from scipy.optimize import fsolve


class Game:

    def __init__(self, maximum_int=10, lin=5, mode=0, c=0):
        self.__mode = mode
        self.__maximum_int = maximum_int
        self.__minimum_int = maximum_int * -1
        self.__c = c
        self.__lin = randrange(2, lin)
        self.__col = self.__lin
        self.__matrix = np.zeros((self.__lin, self.__col))
        self.__matrix2 = np.zeros((self.__lin, self.__col))
        self.fill_matrix(self.__mode, self.__c)

    def getc(self):
        return self.__c

    def setc(self, c):
        self.__c = c

    def delc(self):
        del self.__c

    c = property(getc, setc, delc, "Konstante c für Konstantsummenspiele")

    def getmatrix(self):
        return self.__matrix

    def setmatrix(self, matrix):
        self.__matrix = matrix

    def delmatrix(self):
        del self.__matrix

    matrix = property(getmatrix, setmatrix, delmatrix, "Matrix Spieler 1")

    def getmatrix2(self):
        return self.__matrix2

    def setmatrix2(self, matrix):
        self.__matrix2 = matrix

    def delmatrix2(self):
        del self.__matrix2

    matrix2 = property(getmatrix2, setmatrix2, delmatrix2, "Matrix Spieler 2")

    def getlines(self):
        return self.__lin

    def setlines(self, lin):
        self.__lin = lin
        self.fill_matrix(self.__mode)

    def dellines(self):
        del self.__lin

    lines = property(getlines, setlines, dellines, "Anzahl Zeilen")

    def getcolumns(self):
        return self.__col

    def setcolumns(self, col):
        self.__col = col
        self.fill_matrix(self.__mode)

    def delcolumns(self):
        del self.__col

    cols = property(getcolumns, setcolumns, delcolumns, "Anzahl Spalten")

    def getmode(self):
        return self.__mode

    def setmode(self, mode):
        self.__mode = mode
        self.fill_matrix(mode)

    def delmode(self):
        del self.__mode

    mode = property(getmode, setmode, delmode, "Modus")

    def getmaxint(self):
        return self.__maximum_int

    def setmaxint(self, max_int):
        self.__maximum_int = max_int

    def delmaxint(self):
        del self.__maximum_int

    maxint = property(getmaxint, setmaxint, delmaxint, "Maximal-Wert")

    def getminint(self):
        return self.__minimum_int

    def setminint(self, min_int):
        self.__minimum_int = min_int

    def delminint(self):
        del self.__minimum_int

    minint = property(getminint, setminint, delminint, "Minimal-Wert")

    def fill_matrix(self, mode=0, c=0):
        # Zwei Personen Nullsummenspiele (Modus: 0)
        if mode == 0:
            for count_lin in range(self.__lin):
                for count_col in range(self.__col):
                    x = randrange(self.__minimum_int, self.__maximum_int + 1)
                    self.__matrix[count_lin][count_col] = x
            self.__matrix2 = c - np.asarray(self.__matrix)

        # Nicht kooperative 2-Personenspiele (Modus: 1)
        elif mode == 1:
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
        elif mode == 2:
            for count_lin in range(self.__lin):
                for count_col in range(self.__col):
                    x = randrange(self.__minimum_int, self.__maximum_int + 1)
                    y = randrange(self.__minimum_int, self.__maximum_int + 1)
                    self.__matrix[count_lin][count_col] = x
                    self.__matrix2[count_lin][count_col] = y


new_game = Game()
new_game.matrix = np.asarray([[3, 0, 2],
                                [4,5, 1],
                              [2,2,-1],])
new_game.matrix2 = new_game.matrix*-1

print('solutions')
print(get_calculations_latex(new_game, True))