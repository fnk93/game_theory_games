import numpy as np
from scipy import optimize
from sympy.solvers import solve
from sympy import nsimplify, symbols, Eq
from copy import deepcopy
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patches import Circle, Wedge, Polygon, RegularPolygon
from scipy.spatial import ConvexHull
from pylab import meshgrid

# Lösbar nach Nash:
# Alle Gleichgewichtspunkte sind vertauschbar
# Dominierte Gleichgewichtspunkte möglich


# mode = 0 -> normale Spiele
# mode = 1 -> Kampf der Geschlechter
# TODO: Auszahlungsdiagramm gemischte Strategien für mehr als 2 Stragien nutzbar machen.
def get_payoff_diagramm(payoff_matrix_1, payoff_matrix_2, mode=0):

    #payoff_player_1 = list()
    #payoff_player_2 = list()
    payoff_points = list()
    functions = list()
    for lines in range(payoff_matrix_1.shape[0]):
        for columns in range(payoff_matrix_1.shape[1]):
            #if payoff_matrix_1[lines][columns] not in payoff_player_1 and payoff_matrix_2[lines][columns] not in payoff_player_2:
            #    payoff_player_1.append(payoff_matrix_1[lines][columns])
            #    payoff_player_2.append(payoff_matrix_2[lines][columns])
            if [payoff_matrix_1[lines][columns], payoff_matrix_2[lines][columns]] not in payoff_points:
                payoff_points.append([payoff_matrix_1[lines][columns], payoff_matrix_2[lines][columns]])
                print(payoff_matrix_1[lines][columns], payoff_matrix_2[lines][columns])

    print(payoff_points)
    if mode == 1:
        #outline = ConvexHull(payoff_points, incremental=True)
        payoff_points = np.asarray(payoff_points)

        p = list()
        q = list()
        for lines in range(payoff_matrix_1.shape[0]):
            p.append(np.linspace(0, 1))
        for columns in range(payoff_matrix_2.shape[1]):
            q.append(np.linspace(0,1))

        F3 = 0
        F4 = 0
        for lines in range(payoff_matrix_1.shape[0]):
            for columns in range(payoff_matrix_1.shape[1]):
                print(lines, columns)
                print(payoff_matrix_1[lines][columns])
                if lines > 0 and columns > 0:
                    F3 += payoff_matrix_1[lines][columns] * (1-p[0]) * (1-q[0])
                    F4 += payoff_matrix_2[lines][columns] * (1-p[0]) * (1-q[0])
                elif lines > 0 and columns == 0:
                    F3 += payoff_matrix_1[lines][columns] * (1-p[0]) * q[columns]
                    F4 += payoff_matrix_2[lines][columns] * (1-p[0]) * q[columns]
                elif columns > 0 and lines == 0:
                    F3 += payoff_matrix_1[lines][columns] * p[lines] * (1-q[0])
                    F4 += payoff_matrix_2[lines][columns] * p[lines] * (1-q[0])
                else:
                    F3 += payoff_matrix_1[lines][columns] * p[lines] * q[columns]
                    F4 += payoff_matrix_2[lines][columns] * p[lines] * q[columns]

        return payoff_points, [F3, F4]
    payoff_points.append(payoff_points[0])
    payoff_points = np.asarray(payoff_points)
    print('test', payoff_points)
    return [payoff_points]
    #return payoff_player_1, payoff_player_2, payoff_points

# TODO: Garantiepunkt dominiert in gemischten Strategien
# TODO: Gleichgewichtspunkt zurückgeben mit zugehörigem Strategiepaar(nash und Maximin)
# TODO: Nash-GGW + Wahrscheinlichkeiten + Payoff zusammenführen - done
# TODO: Spielwert-Berechnung bei unterschiedlichen Strategien (Gemischt, rein, Maximin)
# TODO: Maximin bei gemischten Strategien (=Nash-GGW?)
# TODO: Formulierung Lineares Programm ausgliedern aus Simplex
# TODO: Strategie- und Spielwert-Berechnung in Simplex
# TODO: Gleichgewichtspunkte bei Seitenzahlung / keiner Seitenzahlung
# TODO: Graphische Lösung gemischter Strategien


# Leerer Return bedeutet kein Nash-GGW
# Nash-GGW in reinen Strategien
def ggw(payoff_matrix_1, payoff_matrix_2, mode=0):
    optimal = np.zeros((payoff_matrix_1.shape[0], payoff_matrix_1.shape[1]))
    result = list()
    if mode == 0:
        for column in range(payoff_matrix_1.shape[1]):
            max_val_1 = (np.argmax(payoff_matrix_1[:, column]))
            optimal[max_val_1][column] += 1
            for line in range(payoff_matrix_1.shape[0]):
                if line != max_val_1 and payoff_matrix_1[line][column] == payoff_matrix_1[max_val_1][column]:
                    optimal[line][column] += 1
            print(max_val_1)

        for line in range(payoff_matrix_2.shape[0]):
            max_val_2 = (np.argmax(payoff_matrix_2[line]))
            optimal[line][max_val_2] += 1
            for column in range(payoff_matrix_2.shape[1]):
                if column != max_val_2 and payoff_matrix_2[line][column] == payoff_matrix_2[line][max_val_2]:
                    optimal[line][column] += 1
            print(max_val_2)
        print(optimal)
        prep = np.where(optimal == 2)
        for index in range(prep[0].shape[0]):
            result.append([prep[0][index], prep[1][index]])
    return result

O = np.asarray([[3, 2],
                [1, 4]])
F = np.asarray([[2, 3],
                [3, 1]])


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

    for player in range(len(upper_values)):
        determination_intervalls.append([upper_values[player], lower_values[player]])

    return determination_intervalls


# Obere Spielwerte für beide Spieler in reinen Strategien ermitteln
def get_upper_values(payoff_matrix_1, payoff_matrix_2):

    temp_values = list()
    for column in range(payoff_matrix_1.shape[1]):
        temp_values.append(max(payoff_matrix_1[:, column]))
    upper_values = [deepcopy(min(temp_values))]
    temp_values.clear()

    for line in range(payoff_matrix_2.shape[0]):
        temp_values.append(max(payoff_matrix_2[line]))
    upper_values.append(deepcopy(min(temp_values)))
    temp_values.clear()

    return upper_values


# Untere Spielwerte für beide Spieler in reinen Strategien ermitteln
# Spieler 1: Minimum der einzelnen Zeilen, davon das Maximum
# Spieler 2: Minimum der einzelnen Spalten, davon das Maximum
def get_lower_values(payoff_matrix_1, payoff_matrix_2):

    temp_values = list()
    for line in range(payoff_matrix_1.shape[0]):
        temp_values.append(min(payoff_matrix_1[line]))
    lower_values = [deepcopy(max(temp_values))]
    temp_values.clear()

    for columns in range(payoff_matrix_2.shape[1]):
        temp_values.append(min(payoff_matrix_2[:, columns]))
    lower_values.append(deepcopy(max(temp_values)))
    temp_values.clear()

    return lower_values


# Maximin-Strategien der Spieler
# Sollte nur bei determinierten Spielen angewendet werden
def solve_maximin_strategies(payoff_matrix_1, payoff_matrix_2):

    minima_player_1 = list()
    for line in range(payoff_matrix_1.shape[0]):
        minima_player_1.append(np.amin(payoff_matrix_1[line][:]))
    minima_player_2 = list()
    for column in range(payoff_matrix_2.shape[1]):
        minima_player_2.append(np.amin(payoff_matrix_2[:, column]))

    player_1_maximin = list()
    player_2_maximin = list()
    lower_values = get_lower_values(payoff_matrix_1, payoff_matrix_2)
    for strategy in range(len(minima_player_1)):
        if minima_player_1[strategy] == (lower_values[0]):
            player_1_maximin.append(strategy)
    for strategy_2 in range(len(minima_player_2)):
        if minima_player_2[strategy_2] == (lower_values[1]):
            player_2_maximin.append(strategy_2)

    # print('Minmax-Strategien: ', player_1_maximin, player_2_maximin)

    return [player_1_maximin, player_2_maximin]


# Bayes Strategie von player, wenn der andere Spieler strategy wählt
def bayes_strategy(payoff_matrix_1, payoff_matrix_2, player, strategy):

    payoff_matrices = [payoff_matrix_1.transpose(), payoff_matrix_2]
    #print(payoff_matrices[player][strategy])
    bayes = (np.argmax(payoff_matrices[player][strategy], 0))

    return bayes


# Ergebnisse und Lösungswege als PDF formatieren
# TODO: evtl. in Game-Klasse übernehmen
# TODO: Auszahlungsdiagramme graphisch aufbereiten
# TODO: Simplex-Tableaus graphisch aufbereiten
# TODO: Parametrisierung welche Aufgaben gestellt wurden
def get_calculations_pdf(game):

    pass


# Ergebnisse und Lösungswege als LaTeX formatieren
# TODO: evtl. in Game-Klasse übernehmen
# TODO: Auszahlungsdiagramme graphisch aufbereiten
# TODO: Simplex-Tableaus graphisch aufbereiten
# TODO: Parametrisierung welche Aufgaben gestellt wurden
def get_calculations_latex(game):

    pass


# Spielmatrix reduzieren
def reduce_matrix(payoff_matrix_1, payoff_matrix_2):

    # Matrizen für Spieler 1 und 2 müssen betrachtet werden
    reduced_matrix_1 = np.asarray(payoff_matrix_1)
    reduced_matrix_2 = np.asarray(payoff_matrix_2)
    all_compared = False
    run = 0
    reduced = False
    while not all_compared:
        run += 1
        all_compared = True
        dimensions = reduced_matrix_1.shape
        reduce = []
        if dimensions[0] > 2:
            for count in range(dimensions[0]):
                #reducable_line = True
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
                #reducable_column = True
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
def get_strategy_pairs(payoff_matrix_1, payoff_matrix_2):

    strategies = solve_maximin_strategies(payoff_matrix_1, payoff_matrix_2)
    strategy_pairs = list()
    for strategies_player_1 in range(len(strategies[0])):
        for strategies_player_2 in range(len(strategies[1])):
            strategy_pairs.append([strategies[0][strategies_player_1], strategies[1][strategies_player_2]])

    return strategy_pairs


# Punkt aus unteren Spielwerten heißt Garantiepunkt
# Undominiert, wenn kein anderer Auszahlungspunkt existiert bei dem u1 und u2 >= u1* und u2*
# mode = 0 reine Strategien, mode = 1 gemischte Strategien
# TODO: Dominiertheit bei gemischten Strategien erarbeiten
def get_guaranteed_payoff(payoff_matrix_1, payoff_matrix_2, mode=0):
    payoff = list()
    dominated = False
    if mode == 0:
        payoff = get_lower_values(payoff_matrix_1, payoff_matrix_2)
        for lines in range(np.asarray(payoff_matrix_1).shape[0]):
            for columns in range(np.asarray(payoff_matrix_1).shape[1]):
                if payoff_matrix_1[lines][columns] >= payoff[0] and payoff_matrix_2[lines][columns] >= payoff[1]:
                    if payoff_matrix_1[lines][columns] != payoff[0] and payoff_matrix_2[lines][columns] != payoff[1]:
                        dominated = True
    elif mode == 1:
        result = solve_using_nggw(payoff_matrix_1, payoff_matrix_2)
        #print(result)
        for players in range(len(result)):
            #print(result[players][1][-1])
            payoff.append(result[players][0][result[players][1][-1]])

    return payoff, dominated


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
        simplex_game_2 = payoff_matrix_2 - (added_constant_2 + 1)
    else:
        simplex_game_2 = payoff_matrix_2

    return [simplex_game_1, simplex_game_2]


def use_simplex(payoff_matrix_1, payoff_matrix_2):

    simplex_games = make_matrix_ready(payoff_matrix_1, payoff_matrix_2)
    #print('Matrizen: ', simplex_games)
    simplex_1_solution = use_simplex_player1(simplex_games[1])
    simplex_2_solution = use_simplex_player2(simplex_games[0])

    return [simplex_1_solution, simplex_2_solution]


# Simplex-Verfahren für Spieler 1 anwenden
# TODO: Formatierung des Lösungswegs direkt hier machen
def use_simplex_player2(simplex_game_1):

    c = list()
    game_bounds = list()
    A = list()
    b = list()
    for lines in range(np.asarray(simplex_game_1).shape[0]):
        temp = list()
        for columns in range(np.asarray(simplex_game_1.shape[1])):
            temp.append(simplex_game_1[lines][columns])
        A.append(temp)
        b.append(1)

    for columns in range(np.asarray(simplex_game_1.shape[1])):
        c.append(-1)
        game_bounds.append((0, None))

    xk_arr1 = list()
    kwargs_arr1 = list()
    solve_report_1 = SolvingSteps()
    simplex_sol = optimize.linprog(c, A, b, callback=solve_report_1.save_values)
    simplex_steps = solve_report_1.get_array_kwargs()
    simplex_steps_2 = format_solution(solve_report_1.get_array_xk())

    #print('Player 2: ', c, A, b)

    return simplex_sol, simplex_steps, simplex_steps_2


# Simplex-Verfahren für Spieler 2 anwenden
# TODO: Formatierung des Lösungswegs direkt hier machen
def use_simplex_player1(simplex_game_2):

    c = list()
    game_bounds = list()
    A = list()
    b = list()
    for lines in range(np.asarray(simplex_game_2).shape[0]):
        c.append(1)
        game_bounds.append((0, None))

    for columns in range(np.asarray(simplex_game_2.shape[1])):
        b.append(-1)
        temp = list()
        for lines in range(np.asarray(simplex_game_2).shape[0]):
            temp.append(simplex_game_2[lines][columns])
        A.append(temp)

    xk_arr2 = list()
    kwargs_arr2 = list()
    solve_report2 = SolvingSteps()
    simplex_sol = optimize.linprog(c, A, b, callback=solve_report2.save_values)
    simplex_steps = solve_report2.get_array_kwargs()
    simplex_steps_2 = format_solution(solve_report2.get_array_xk())

    #print('Player 1: ', c, A, b)

    return simplex_sol, simplex_steps, simplex_steps_2


# Lösung mit Bedingungen für NGGW
# Gemischte Maximin-Strategien der Spieler
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
    w2 = symbols('w', real=True)
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
    #print(solution)

    return solution


def format_solution(solution_array):

    #print('Formatting: ')
    solution = list()
    #print(np.asarray(solution_array))
    #print('Formatting:')
    #print(np.asarray(solution_array))
    for line in range(np.asarray(solution_array).shape[0]):
        temp = list()
        for column in range(np.asarray(solution_array).shape[1]):
            temp.append(nsimplify(solution_array[line][column], tolerance=0.0001, rational=True))
            #print(nsimplify(solution_array[line][column], tolerance=0.0001, rational=True))
        if np.amin(temp) != 0 or np.amax(temp) != 0:
            solution.append(temp)
    #print(solution)

    return solution

# Callable Methode um Zwischenschritte des Simplex abzufangen
class SolvingSteps:


    def __init__(self):
        self.__array_xk = []
        self.__array_kwargs = []
        #print('Initialisiert mit:')
        #print('xk:')
        #print(self.__array_xk)
        #print('kwargs:')
        #print(self.__array_kwargs)

    def save_values(self, xk, **kwargs):
        #print('Speichern:')
        #print(xk)
        #print(self.__array_xk)
        #for i in range(len(self.__array_xk)):
        #    print(self.__array_xk[i])
        #self.array_xk = xk
        #print('xk:')
        #print(np.asarray(xk))
        #print('array xk:')
        #print(self.get_array_xk())
        temp_values = {}
        for key, value in kwargs.items():
            temp_values[key] = deepcopy(value)
            #print('Key-Value-Paar:')
            #print(key, value)
        #print(temp_values['tableau'])
        self.__array_kwargs.append(deepcopy(temp_values))
        #print('nach kwargs:')
        #print(self.__array_xk)
        #print(xk)
        self.__array_xk.append(deepcopy(xk))
        #print('added:')
        #print(xk)
        #print(self.__array_xk)

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

print(B[:,1])

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
print(D[0])
print(sol[0])
print(sol[1])

E = np.asarray([[0, -1, 2],
                [2, 0, -1],
                [-1, 2, 0]])
simplex = (use_simplex(E, E*-1))
print()
print('Fun: ')
print(simplex[0][0]["fun"])
print('X: ')
print(simplex[0][0]['x'])
print()
print('Fun: ')
print(simplex[1][0]["fun"])
print('X: ')
print(simplex[1][0]['x'])
print()
for table in range(len(simplex[1][1])):
    print('Tableau ', table)
    print(np.asarray(format_solution(simplex[1][1][table]["tableau"])))

print(simplex[1][0])

for table in range(len(simplex[0][1])):
    print('Tableau ', table)
    print(np.asarray(format_solution(simplex[0][1][table]["tableau"])))

print(simplex[0][0])

F = np.asarray([[2, -1],
                [-1, 1]])

G = np.asarray([[1, -1],
                [-1, 2]])

print(solve_using_nggw(F, G)[0][1])
for val in solve_using_nggw(F, G)[0][1]:
    print(val, solve_using_nggw(F, G)[0][0][val])

H = np.asarray([[1, -2],
               [-1, 1]])


player = 1
other_strategy = 0
print('Bayes Strategie für Spieler ', player+1, ' gegenüber Strategie ', other_strategy+1, ' von Spieler 1:')
print(bayes_strategy(H, H*-1, player, other_strategy) + 1)

U = np.asarray([[10, 1],
                [0, 1000],
                [2000, 0]])

print(bayes_strategy(U, U*-1, 0, 1) + 1)

R = np.asarray([[4, 5, 2],
                [6, 3, 2]])

print(get_strategy_pairs(R, R*-1))
print(is_determined(R, R*-1))

S = np.asarray([[0, -1, 2],
                [2, 0, -1],
                [-1, 2, 0]])

coords = get_payoff_diagramm(S, S*-1)

T = np.asarray([[2, -1],
                [-1, 1]])
U = np.asarray([[1, -1],
                [-1, 2]])

fig = plt.figure()
ax = fig.add_subplot(111)
coords = get_payoff_diagramm(T, U, 1)
print(coords)
plt.ylabel('Auszahlung Spieler 2')
plt.xlabel('Auszahlung Spieler 1')

#coords_2 = list()
#for x in range(len(coords[0])):
#    coords_2.append([coords[0][x], coords[1][x]])

#coords_2 = np.asarray(coords_2)
#print(coords_2)
print(coords)
#points = np.random.rand(30, 2)
#print(points)
#hull = ConvexHull(coords)
#print(hull)

p1 = np.linspace(0, 1)
p2 = np.linspace(0, 1)
q1 = np.linspace(0, 1)
q2 = np.linspace(0, 1)

# x-Werte
F1 = 2*p1*q1 - 1*p1*q2 - 1*p2*q1 + 1*p2*q2
#F3 = 2*p1*q1 - 1*p1*(1-q1) - 1*(1-p1)*q1 + 1*(1-p1)*(1-q1)
# y-Werte
F2 = 1*p1*q1 - 1*p1*q2 - 1*p2*q1 + 2*p2*q2
#F4 = 1*p1*q1 - 1*p1*(1-q1) - 1*(1-p1)*q1 + 2*(1-p1)*(1-q1)

#print(len(F3))
#print(F3)
#print(len(F4))
#print(F4)
#plt.plot(F1, F2, 'k-')

#print(coords[1][0])
#print(coords[1][1])

new_points = list()
#for points in range(len(F3)):
#    new_points.append((np.asarray([F3[points], F4[points]])))

new_points = np.asarray(new_points)
print(new_points)

#coords[1].add_points(new_points)
#print(coords[1].points)
#for simplex in coords[1].simplices:
#    plt.plot(coords[0][simplex, 0], coords[0][simplex, 1], 'k-')



#plt.plot(coords[0][coords[1].vertices[0],0], coords[0][coords[1].vertices[0],1], 'ro')



P1, P2, Q1, Q2 = meshgrid(p1, p2, q1, q2)

FF1 = func(P1, P2, Q1, Q2)
FF2 = func_2(P1, P2, Q1, Q2)

#print(FF1)
print(coords[0])
#plt.plot(F3, F4, 'k-')
plt.plot(coords[1][0], coords[1][1], 'k-')
print('test')
x = list()
y = list()
for points in range(len(coords[0])):
    print(coords[0][points])
    print(points)
    x.append(coords[0][points][0])
    y.append(coords[0][points][1])
    #plt.plot(points[0], points[1], 'k-')
    #print(points)
    #print(coords[points])
    #print(coords[points][:, 0], coords[0][points][:, 1])

#x.append(coords[0][0])
#y.append(coords[0][1])
plt.plot(x, y, 'k-')
#plt.plot(coords_2[hull.vertices,0], coords_2[hull.vertices,1], 'r--', lw=2)
#plt.plot(coords_2[hull.vertices[0],0], coords_2[hull.vertices[0],1], 'ro')
plt.axis([-5, 5, -5, 5])
plt.show()

#polygon = Polygon(coords_2, True, joinstyle='bevel')

#ax.add_patch(polygon)

#ax.set_xlim(-15,15)
#ax.set_ylim(-15,15)

#plt.show()


#plt.plot(polygon)
#plt.axis([-15, 15, -15, 15])
#plt.grid(True)
#plt.show()

Z = np.asarray([[5, 4, 3],
                [3, 2, 1],
                [2, 1, 0]])

solve_maximin_strategies(Z, Z*-1)

print(get_guaranteed_payoff(T, U, 0))
print(solve_using_nggw(T, U))

TT = np.asarray([[1, -1],
                 [-1, 1]])

print(get_guaranteed_payoff(TT, TT*-1, 0))