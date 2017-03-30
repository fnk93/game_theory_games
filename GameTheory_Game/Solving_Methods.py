import numpy as np
from scipy import optimize
from sympy.solvers import solve
from sympy import nsimplify, symbols, Eq


# Prüft ob für jeden Spieler unterer Spielwert dem oberen entspricht
def is_determined(payoff_matrix_1, payoff_matrix_2):
    det_intervalls = determination_intervall(payoff_matrix_1, payoff_matrix_2)

    for i in range(len(det_intervalls)):
        if min(det_intervalls[i]) != max(det_intervalls[i]):
            return False
    return True


# Determiniertheitsintervall für beide Spieler berechnen
def determination_intervall(payoff_matrix_1, payoff_matrix_2):
    upper_values = get_upper_values(payoff_matrix_1, payoff_matrix_2)
    lower_values = get_lower_values(payoff_matrix_1, payoff_matrix_2)
    determination_intervalls = list()

    for i in range(len(upper_values)):
        determination_intervalls.append([upper_values[i], lower_values[i]])

    return determination_intervalls


# Obere Spielwerte für beide Spieler ermitteln
def get_upper_values(payoff_matrix_1, payoff_matrix_2):
    temp_values = list()
    for i in range(payoff_matrix_1.transpose().shape[0]):
        temp_values.append(max(payoff_matrix_1.transpose()[i]))
    upper_values = [min(temp_values)]
    temp_values.clear()

    for i in range(payoff_matrix_2.shape[0]):
        temp_values.append(max(payoff_matrix_2[i]))
    upper_values.append(min(temp_values))
    temp_values.clear()
    return upper_values


# Untere Spielwerte für beide Spieler ermitteln
def get_lower_values(payoff_matrix_1, payoff_matrix_2):
    temp_values = list()
    for i in range(payoff_matrix_1.shape[0]):
        temp_values.append(min(payoff_matrix_1[i]))
    lower_values = [max(temp_values)]
    temp_values.clear()

    for i in range(payoff_matrix_2.transpose().shape[0]):
        temp_values.append(min(payoff_matrix_2.transpose()[i]))
    lower_values.append(max(temp_values))
    temp_values.clear()
    return lower_values


# Maximin-Strategien der Spieler
# Sollte nur bei determinierten Spielen angewendet werden
def solve_maximin_strategies(payoff_matrix_1, payoff_matrix_2):
    det_int = determination_intervall(payoff_matrix_1, payoff_matrix_2)
    temp_strategies = list()
    for i in range(payoff_matrix_1.shape[0]):
        if min(det_int[0]) == min(payoff_matrix_1[i]):
            temp_strategies.append(i)
    maximin_strategies = [temp_strategies]
    temp_strategies.clear()

    for i in range(payoff_matrix_2.transpose().shape[0]):
        if min(det_int[1]) == min(payoff_matrix_2.transpose()[i]):
            temp_strategies.append(i)
    maximin_strategies.append(temp_strategies)
    return maximin_strategies


# Ergebnisse und Lösungswege als PDF formatieren
# TODO: evtl. in Game-Klasse übernehmen
def get_calculations_pdf(game):
    pass


# Ergebnisse und Lösungswege als LaTeX formatieren
# TODO: evtl. in Game-Klasse übernehmen
def get_calculations_latex(game):
    pass


# Spielmatrix reduzieren
def reduce_matrix(game):
    pass


# Matrix aller MinMax-Strategie-Paare ausgeben
# TODO: evtl. in Game-Klasse übernehmen
def get_strategy_pairs(game):
    pass


# Simplex-Verfahren für Spieler 1 anwenden
# TODO: Formatierung des Lösungswegs direkt hier machen
def use_simplex_player1(game):
    pass


# Simplex-Verfahren für Spieler 2 anwenden
# TODO: Formatierung des Lösungswegs direkt hier machen
def use_simplex_player2(game):
    pass


# Lösung mit Bedingungen für NGGW
def solve_using_nggw(game):
    pass


# Callable Methode um Zwischenschritte des Simplex abzufangen
class SolvingSteps:

    def __init__(self):
        self.__array_xk = []
        self.__array_kwargs = []

    # TODO: für jedes Key - Value Paar aus kwargs Ergebnisse speichern
    def __call__(self, xk, **kwargs):
        self.__array_xk.append(xk)
        self.__array_kwargs.append(kwargs['tableau'])

    def get_array_kwargs(self):
        return self.__array_kwargs

    def get_array_xk(self):
        return self.__array_xk


# Kleine Tests der Funktionen
A = np.asarray([[1, 2, 3],
                [0, 1, 2]])
B = np.asarray([[-1, -2, -3],
                [0, -1, -2]])
print(is_determined(A, B))
print(solve_maximin_strategies(A, B))
