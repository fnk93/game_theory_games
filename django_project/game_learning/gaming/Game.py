import numpy as np


class Game:

    def __init__(self, maximum_int=10, minimum_int=-10, lin=np.random.randint(2, 5), col=np.random.randint(2, 5)):
        self.__maximum_int = maximum_int
        self.__minimum_int = minimum_int
        self.__lin = lin
        self.__col = col
        self.__matrix = np.zeros((self.__lin, self.__col))
        self.__matrix2 = np.zeros((self.__lin, self.__col))
        self.fill_matrix()

    def getmatrix(self):
        return self.__matrix

    def setmatrix(self, matrix):
        self.__matrix = matrix

    matrix = property(getmatrix, setmatrix, None, "Matrix Spieler 1")

    def getmatrix2(self):
        return self.__matrix2

    def setmatrix2(self, matrix):
        self.__matrix2 = matrix

    matrix2 = property(getmatrix2, setmatrix2, None, "Matrix Spieler 2")

    def getlines(self):
        return self.__lin

    def setlines(self, lin):
        self.__lin = lin
        self.fill_matrix()

    lines = property(getlines, setlines, None, "Anzahl Zeilen")

    def getcolumns(self):
        return self.__col

    def setcolumns(self, col):
        self.__col = col
        self.fill_matrix()

    cols = property(getcolumns, setcolumns, None, "Anzahl Spalten")

    def getmaxint(self):
        return self.__maximum_int

    def setmaxint(self, max_int):
        self.__maximum_int = max_int
        self.fill_matrix()

    maxint = property(getmaxint, setmaxint, None, "Maximal-Wert")

    def getminint(self):
        return self.__minimum_int

    def setminint(self, min_int):
        self.__minimum_int = min_int
        self.fill_matrix()

    minint = property(getminint, setminint, None, "Minimal-Wert")

    def fill_matrix(self):
        # Nicht kooperative 2-Personenspiele
        self.matrix = np.random.randint(self.minint, self.maxint+1, size=(self.lines,self.cols))
        self.matrix2 = np.random.randint(self.minint, self.maxint+1, size=(self.lines,self.cols))