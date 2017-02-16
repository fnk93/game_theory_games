import numpy as np
from random import randrange
from scipy import optimize

# TODO: Determiniertheit, Maximin-Strategien, gemischte Strategien + Lösungswege


class Solve(object):

    def __init__(self, game):
        self.__matrix = game.get_matrix()
        self.__determined = False
        self.__determinedIntervall = []
        self.__top_value2 = 0
        self.__top_value1 = 0
        self.is_determined()
        self.__maximin_strategies1 = []
        self.__maximin_strategies2 = []
        self.solve_strategies()
        self.__reduced_matrix = []
        self.__solutions = []
        self.solving_array()
        self.__reduced = False
        if not self.__determined:
            self.reduce_matrix()


    # Determiniertheit des Spiels bestimmen
    # Determiniertheitsintervall berechnen
    def is_determined(self):
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

    # Ermittlung der Maximin-Strategien beider Spieler
    def solve_strategies(self):
        for count in range(self.__matrix.shape[0]):
            if min(self.__matrix[count]) == self.__top_value1:
                self.__maximin_strategies1.append(count+1)

        for count in range(self.__matrix.transpose().shape[0]):
            if max(self.__matrix.transpose()[count]) == self.__top_value2:
                self.__maximin_strategies2.append(count+1)

    # Maximin-Strategien-Array für Spieler 1
    def get_maximin1(self):
        return self.__maximin_strategies1

    # Maximin-Strategien-Array für Spieler 2
    def get_maximin2(self):
        return self.__maximin_strategies2

    # Ist das Spiel determiniert?
    def get_determined(self):
        return self.__determined

    # Determiniertheitsintervall
    def get_determined_intervall(self):
        return self.__determinedIntervall

    # Unterer Spielwert
    def get_low_value(self):
        return self.__top_value1

    # Oberer Spielwert
    def get_high_value(self):
        return self.__top_value2

    # Funktion um Ergebnisse in Textform zu präsentieren
    def output(self):
        if self.__determined:
            print('Spiel ist determiniert mit Spielwert: ' + str(self.__top_value1))
            print('Auszahlung für Spieler 1: ' + str(self.__top_value1))
            print('Auszahlung für Spieler 2: ' + str(-1 * self.__top_value2))
        else:
            print('Spiel ist nicht determiniert mit Indeterminiertheitsintervall: ' + str(self.__determinedIntervall))
        print('Maximin-Strategie(n) von Spieler 1: ' + str(self.__maximin_strategies1))
        print('Maximin-Strategie(n) von Spieler 2: ' + str(self.__maximin_strategies2))
        print('Strategiekombinationen: ' + str(self.__solutions))
        if self.__reduced:
            print('Reduziertes Spiel: \n' + str(self.__reduced_matrix))

    # Matrix reduzieren falls möglich
    def reduce_matrix(self):
        reduced_matrix = np.asarray(self.__matrix)
        all_compared = False
        while not all_compared and (reduced_matrix.shape[0] >= 2 and reduced_matrix.shape[1] >= 2):
            all_compared = True
            dimensions = reduced_matrix.shape
            reduce = []
            for count in range(dimensions[0]):
                reducable_line = True
                added = False
                for count_2 in range(dimensions[0]):
                    reducable_line = True
                    if count != count_2:
                        for count_3 in  range(dimensions[1]):
                            if reduced_matrix[count][count_3] > reduced_matrix[count_2][count_3] and reducable_line:
                                reducable_line = False
                        if reducable_line:
                            if not added:
                                reduce.append(count)
                                all_compared = False
                                added = True
            i = 0
            for count in range(len(reduce)):
                if reduced_matrix.shape[0] > 2:
                    reduced_matrix = np.delete(reduced_matrix, reduce[count]-i, 0)
                    i += 1
                    self.__reduced = True
            print(reduce)
            dimensions = reduced_matrix.shape
            reduce = []

            for count in range(dimensions[1]):
                reducable_column = True
                added = False
                for count_2 in range(dimensions[1]):
                    reducable_column = True
                    if count != count_2:
                        for count_3 in range(dimensions[0]):
                            if reduced_matrix[count_3][count] < reduced_matrix[count_3][count_2] and reducable_column:
                                reducable_column = False
                        if reducable_column:
                            if not added:
                                reduce.append(count)
                                all_compared = False
                                added = True
            i = 0
            for count in range(len(reduce)):
                if reduced_matrix.shape[1] > 2:
                    reduced_matrix = np.delete(reduced_matrix, reduce[count] - i, 1)
                    i += 1
                    self.__reduced = True
            print(reduce)
        self.__reduced_matrix = reduced_matrix


    # Lösungsarray
    def solving_array(self):
        for count in range(np.asarray(self.__maximin_strategies1).shape[0]):
            for count_2 in range(np.asarray(self.__maximin_strategies2).shape[0]):
                self.__solutions.append([self.__maximin_strategies1[count], self.__maximin_strategies2[count_2]])

