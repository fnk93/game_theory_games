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
    det_intervalls = determination_intervall(payoff_matrix_1, payoff_matrix_2)
    temp_strategies = list()
    for i in range(payoff_matrix_1.shape[0]):
        if min(det_intervalls[0]) == min(payoff_matrix_1[i]):
            temp_strategies.append(i)
    maximin_strategies = [temp_strategies]
    temp_strategies.clear()

    for i in range(payoff_matrix_2.transpose().shape[0]):
        if min(det_intervalls[1]) == min(payoff_matrix_2.transpose()[i]):
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
def reduce_matrix(payoff_matrix_1, payoff_matrix_2):

    # Matrizen für Spieler 1 und 2 müssen betrachtet werden
    reduced_matrix_1 = np.asarray(payoff_matrix_1)
    reduced_matrix_2 = np.asarray(payoff_matrix_2)
    all_compared = False
    run = 0
    while not all_compared:
        run += 1
        all_compared = True
        dimensions = reduced_matrix_1.shape
        reduce = []
        if dimensions[0] > 2:
            for count in range(dimensions[0]):
                reducable_line = True
                added = False
                for count_2 in range(dimensions[0]):
                    reducable_line = True
                    if count != count_2:
                        for count_3 in range(dimensions[1]):
                            if reduced_matrix_1[count][count_3] > reduced_matrix_1[count_2][count_3] and reducable_line:
                                reducable_line = False
                        if reducable_line:
                            if not added:
                                reduce.append(count)
                                all_compared = False
                                added = True
            i = 0
            for count in range(len(reduce)):
                if dimensions[0] > 2:
                    reduced_matrix_1 = np.delete(reduced_matrix_1, reduce[count] - i, 0)
                    reduced_matrix_2 = np.delete(reduced_matrix_2, reduce[count] - i, 0)
                    i += 1
                    reduced = True
                    dimensions = reduced_matrix_1.shape
            dimensions = reduced_matrix_1.shape
        reduce = []

        if dimensions[1] > 2:
            for count in range(dimensions[1]):
                reducable_column = True
                added = False
                for count_2 in range(dimensions[1]):
                    reducable_column = True
                    if count != count_2:
                        for count_3 in range(dimensions[0]):
                            if reduced_matrix_2[count_3][count] > reduced_matrix_2[count_3][count_2] and reducable_column:
                                reducable_column = False
                        if reducable_column:
                            if not added:
                                reduce.append(count)
                                all_compared = False
                                added = True
            i = 0
            for count in range(len(reduce)):
                if dimensions[1] > 2:
                    reduced_matrix_1 = np.delete(reduced_matrix_1, reduce[count] - i, 1)
                    reduced_matrix_2 = np.delete(reduced_matrix_2, reduce[count] - i, 1)
                    i += 1
                    reduced = True
                    dimensions = reduced_matrix_1.shape
        if reduced_matrix_1.shape[0] <= 2 and reduced_matrix_1.shape[1] <= 2:
            all_compared = True

    return [reduced_matrix_1, reduced_matrix_2, reduced]


# Matrix aller MinMax-Strategie-Paare ausgeben
# TODO: evtl. in Game-Klasse übernehmen
def get_strategy_pairs(game):
    pass


# Matrizen für Simplex vorbereiten
# Konstante hinzurechnen, sodass Spieler1-Matrix absolut positiv und Spieler2-Matrix absolut negativ ist
def make_matrix_ready(payoff_matrix_1, payoff_matrix_2):

    # Matrix-Minimum für Spieler 1 herausfinden
    # Matrix-Maximum für Spieler 2 herausfinden
    added_constant_1 = np.amin(payoff_matrix_1)
    added_constant_2 = np.amax(payoff_matrix_2)

    # Matrix von Spieler 1 wird in absolut positive Matrix überführt
    if added_constant_1 < 1:
        simplex_game_1 = payoff_matrix_1 - (added_constant_1 - 1)
    else:
        simplex_game_1 = payoff_matrix_1

    # Matrix von Spieler 2 wird in absolut negative Matrix überführt
    if added_constant_2 > -1:
        simplex_game_2 = payoff_matrix_2 - (added_constant_1 + 1)
    else:
        simplex_game_2 = payoff_matrix_2

    return [simplex_game_1, simplex_game_2]


# Simplex-Verfahren für Spieler 1 anwenden
# TODO: Formatierung des Lösungswegs direkt hier machen
def use_simplex_player1(game):
    pass


# Simplex-Verfahren für Spieler 2 anwenden
# TODO: Formatierung des Lösungswegs direkt hier machen
def use_simplex_player2(game):
    pass


# Lösung mit Bedingungen für NGGW
def solve_using_nggw(payoff_matrix_1, payoff_matrix_2):
    # Gemischte Strategien p für Spieler 1 und Spielwert w für Spieler 2

    # Variablen des LGS deklarieren
    p = symbols('p:' + str(payoff_matrix_2.shape[0]), nonnegative=True)
    w = symbols('w', real=True)
    # Lösungssystem erstellen und mit Gleichungen füllen
    # Variablensystem erstellen und füllen
    u = list()
    for column in range(payoff_matrix_2.shape[1]):
        temp = 0
        for line in range(payoff_matrix_2.shape[0]):
            temp += payoff_matrix_2[line][column]*p[line]
        u.append(Eq(temp, w))
    temp_2 = 0
    symbol = list()
    for decisions in range(len(p)):
        temp_2 += 1*p[decisions]
        symbol.append(p[decisions])
    u.append(Eq(temp_2, 1))
    symbol.append(w)

    # LGS lösen und speichern für Rückgabe
    solution_1 = solve(u, force=True)
    solution = list()
    solution.append([solution_1, symbol])

    # Gemischte Strategien q für Spieler 2 und Spielwert w für Spieler 1

    # Variablen des LGS deklarieren
    q = symbols('q:' + str(payoff_matrix_1.shape[1]), nonnegative=True)
    w2 = symbols('w2', real=True)
    # Lösungssystem erstellen und mit Gleichungen füllen
    # Variablensystem erstellen und füllen
    u2 = list()
    for line in range(payoff_matrix_1.shape[0]):
        temp = 0
        for column in range(payoff_matrix_1.shape[1]):
            temp += payoff_matrix_1[line][column]*q[column]
        u2.append(Eq(temp, w2))
    temp_2 = 0
    symbol = list()
    for decisions in range(len(q)):
        temp_2 += 1*q[decisions]
        symbol.append(q[decisions])
    u2.append(Eq(temp_2, 1))
    symbol.append(w2)

    # LGS lösen und speichern für Rückgabe
    solution_2 = solve(u2, force=True)
    solution.append([solution_2, symbol])

    # solution[0] enthält Spielwert für Spieler 2 und Strategien für Spieler 1 und die zugehörigen Variablen
    # solution[1] enthält Spielwert für Spieler 1 und Strategien für Spieler 2 und die zugehörigen Variablen
    print(solution)
    return solution


# Callable Methode um Zwischenschritte des Simplex abzufangen
class SolvingSteps:

    def __init__(self):
        self.__array_xk = []
        self.__array_kwargs = []

    def __call__(self, xk, **kwargs):
        self.__array_xk.append(xk)
        temp_values = {}
        for key, value in kwargs.items():
            temp_values[key] = value
        self.__array_kwargs.append(temp_values)

    # Funktion um Dictionaries auszugeben
    def get_array_kwargs(self):
        return self.__array_kwargs

    # Funktion um Parameter des Simplex abzufragen
    def get_array_xk(self):
        return self.__array_xk


# Kleine Tests der Funktionen
A = np.asarray([[1, 2, 3],
                [0, 1, 2]])
B = np.asarray([[-1, -2, -3],
                [0, -1, -2]])

# Determiniertheit und Maximin testen
print(is_determined(A, B))
print(solve_maximin_strategies(A, B))

# Tests zur Matrixtransformation
C = np.asarray([[0, -1, 2],
                [2, 0, -1],
                [-1, 2, 0]])
print(C)
print(np.rot90(C, 2))
print(np.fliplr(np.flipud(C)))
print(C*-1)

# Spiel mit NGGW-Bedingung lösen
solve_using_nggw(C, C*-1)
D = np.asarray([[4, 1, 8, 0],
                [5, 2, 2, 1],
                [10, 2, 7, 8],
                [-6, 5, 6, 2]])

# Matrix reduzieren
sol = (reduce_matrix(D, D*-1))
print(sol[0])
print(sol[1])
