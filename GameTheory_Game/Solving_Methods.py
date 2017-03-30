import numpy as np
from scipy import optimize
from sympy.solvers import solve
from sympy import nsimplify, symbols, Eq

# Spiel determiniert?
def is_determined(game):
    pass

# Determiniertheitsintervall berechnen
def determination_intervall(game):
    pass

# Obere Spielwerte
def get_upper_values(game):
    pass

# Untere Spielwerte
def get_lower_values(game):
    pass

# Maximin-Strategien der Spieler
def solve_minmax_strategies(game):
    pass

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
class SolvingSteps():

    def __init__(self):
        self.__array_xk = []
        self.__array_kwargs = []

    # TODO: für jedes Key - Value Paar aus kwargs Ergebnisse speichern
    def __call__(self, xk, **kwargs):
        self.__array_xk.append(xk)
        self.__array_kwargs.append(kwargs['tableau'])

    def getArrayKwargs(self):
        return self.__array_kwargs

    def getArrayXk(self):
        return self.__array_xk
