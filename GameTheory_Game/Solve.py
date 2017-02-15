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
        else:
            print('Spiel ist nicht determiniert mit Indeterminiertheitsintervall: ' + str(self.__determinedIntervall))
        print('Maximin-Strategie(n) von Spieler 1: ' + str(self.__maximin_strategies1))
        print('Maximin-Strategie(n) von Spieler 2: ' + str(self.__maximin_strategies2))
