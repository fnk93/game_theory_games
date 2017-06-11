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
        self.fill_matrix()

    def dellines(self):
        del self.__lin

    lines = property(getlines, setlines, dellines, "Anzahl Zeilen")

    def getcolumns(self):
        return self.__col

    def setcolumns(self, col):
        self.__col = col
        self.fill_matrix()

    def delcolumns(self):
        del self.__col

    cols = property(getcolumns, setcolumns, delcolumns, "Anzahl Spalten")

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

    def fill_matrix(self):
        # Nicht kooperative 2-Personenspiele
        self.matrix = np.random.randint(self.minint, self.maxint+1, size=(self.lines,self.cols))
        self.matrix2 = np.random.randint(self.minint, self.maxint+1, size=(self.lines,self.cols))