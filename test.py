from scipy.optimize import linprog
import numpy as np
from sympy import nsimplify
import matplotlib


# Funktion für Zwischenschritte
# TODO: Aufbereiten und nach verschiedenen Lösungswegen suchen.
class Tableau(object):
    def __call__(self, xk, **kwargs):
        print('Aktuelle Lösung: ' + str(xk))
        print(kwargs['tableau'])
        print(kwargs['pivot'])
        print(kwargs['basis'])
#        for key in kwargs:
#            print(key)
#            print(kwargs[key])


# Ermittlung Oberer + Unterer Spielwert in reinen Strategien

# matrix = [[4, 5, 2],
#          [6, 3, 2]]

# matrix = [[1, -1],
#          [-1, 1]]

# matrix = [[4, 1, 8, 0],
#          [5, 2, 2, 1],
#          [10, 2, 7, 8],
#          [-6, 5, 6, 2]]

matrix = [[2, 1, 4],
          [4, 2, 1],
          [1, 4, 2]]

# matrix = [[4, 1, 2],
#          [1, 5, 0],
#          [4, 3, 3]]

test_game = np.asarray(matrix)
print(test_game)

dimensionen_game = test_game.shape
lines_game = dimensionen_game[0]
columns_game = dimensionen_game[1]

test_game_transposed = test_game.transpose()
print(test_game_transposed)

dimensionen_game_transposed = test_game_transposed.shape
lines_game_transposed = dimensionen_game_transposed[0]
columns_game_transposed = dimensionen_game_transposed[1]

# Oberer Spielwert: min (j) max (i) u_ij
max_player_2 = []

for count in range(lines_game_transposed):
    print(str(test_game_transposed[count]) + ', Spaltennmaximum: ' + str(max(test_game_transposed[count])))
    max_player_2.append(max(test_game_transposed[count]))

# print(test_game[0])
# print(max(test_game[0]))
# print(test_game[1])
# print(max(test_game[1]))
# max_player_2.append(max(test_game[0]))
# max_player_2.append(max(test_game[1]))

top_value_2 = min(max_player_2)
print('Oberer Spielwert: ' + str(top_value_2))


# Unterer Spielwert: max (i) min (j) u_ij
max_win_player_1 = []

for count in range(lines_game):
    print(str(test_game[count]) + ', Zeilenminimum: ' + str(min(test_game[count])))
    max_win_player_1.append(min(test_game[count]))

# print(test_game_transposed[0])
# print(min(test_game_transposed[0]))
# print(test_game_transposed[1])
# print(min(test_game_transposed[1]))
# max_win_player_1.append(min(test_game_transposed[0]))
# max_win_player_1.append(min(test_game_transposed[1]))

top_value_1 = max(max_win_player_1)
print('Unterer Spielwert: ' + str(top_value_1))

determined = False
indetermined_intervall = []
if top_value_1 != top_value_2:
    print('Spiel ist indeterminiert.')
    indetermined_intervall.append(top_value_1)
    indetermined_intervall.append(top_value_2)
    print('Indeterminiertheitsintervall: ' + str(indetermined_intervall))
else:
    print('Spiel ist determiniert.')
    determined = True

# Maximin-Strategien Spieler 1
strategies_1 = []
for count in range(lines_game):
    if min(test_game[count]) == top_value_1:
        strategies_1.append(count+1)

# Maximin-Strategien Spieler 2
strategies_2 = []
for count in range(lines_game_transposed):
    if max(test_game_transposed[count]) == top_value_2:
        strategies_2.append(count+1)

print('Maximin-Strategien Spieler 1: ' + str(strategies_1))
print('Maximin-Strategien Spieler 2: ' + str(strategies_2))

# Lösungen des Spiels
solutions = []
for count in range(np.asarray(strategies_1).shape[0]):
    for count_2 in range(np.asarray(strategies_2).shape[0]):
        solutions.append([strategies_1[count], strategies_2[count_2]])

print(solutions)

# Indeterminiertes Spiel mit gemischten Strategien lösen
# Summe der Wahrscheinlichkeiten p und q muss 1 ergeben
if not determined:
    p = np.zeros(lines_game)
    q = np.zeros(columns_game)
    print(p)
    print(q)
    reduced_game = np.asarray(matrix)

    allCompared = False
    while not allCompared and (reduced_game.shape[0] > 2 and reduced_game.shape[1] > 2):

        # Annahme, dass falls kein Löschen möglich ist nicht mehr verglichen werden muss.
        allCompared = True
        dimensionen_reduced = reduced_game.shape
        reduced_lines = dimensionen_reduced[0]
        reduced_columns = dimensionen_reduced[1]
        # Lösch-Array leeren
        reduce = []
        # Überprüfung der Zeilen auf dominierte Zeilen und Sammeln eben dieser.
        for count in range(reduced_lines):
            reducable_line = True
            for count_2 in range(reduced_lines):
                reducable_line = True
                if count != count_2:
                    for count_3 in range(reduced_columns):
                        if reduced_game[count][count_3] > reduced_game[count_2][count_3] and reducable_line:
                            reducable_line = False
                    if reducable_line:
                        reduce.append(count)
                        allCompared = False


        # Reduzierung der dominierten Zeilen
        i = 0
        for count in range(len(reduce)):
            reduced_game = np.delete(reduced_game, reduce[count]-i, 0)
            i += 1

        # Dimensionen neu zuweisen, da evtl. Zeilen gelöscht
        print(reduced_game)
        dimensionen_reduced = reduced_game.shape
        reduced_lines = dimensionen_reduced[0]
        reduced_columns = dimensionen_reduced[1]

        # Lösch-Array leeren
        reduce = []
        # Überprüfung der Spalten auf dominierte Spalten und Sammeln eben dieser.
        for count in range(reduced_columns):
            reducable_column = True
            for count_2 in range(reduced_columns):
                reducable_column = True
                if count != count_2:
                    for count_3 in range(reduced_lines):
                        if reduced_game[count_3][count] < reduced_game[count_3][count_2] and reducable_column:
                            reducable_column = False
                    if reducable_column:
                        reduce.append(count)
                        allCompared = False


        # Reduzierung der dominierten Spalten
        i = 0
        for count in range(len(reduce)):
            reduced_game = np.delete(reduced_game, reduce[count] - i, 1)
            i += 1
        print(reduced_game)

    # Vorbereiten des Simplex Algorithmus'
    # Zielfunktion zum minimieren x1 + x2 + x3
    c = []
    for count in range(reduced_game.shape[1]):
        c.append(1)

    # Nebenbedingungen in Matrix-Form
    A = []
    for count in range(reduced_game.shape[1]):
        temp = []
        for count_2 in range(reduced_game.shape[0]):
            temp.append(reduced_game[count_2][count]*-1)
        A.append(temp)

    # Beschränkungen der Nebenbedingungen
    b = []
    for count in range(reduced_game.shape[1]):
        b.append(-1)

    # Beschränkungen für Variablen x1, x2, ..., xn
    bounds_here = []
    for count in range(len(c)):
        bounds_here.append((0, None))

    # Durchführen des Simplex, Ergebnis beinhaltet Lösungsvektor x
    result = linprog(c, A, b, bounds=bounds_here, options={"disp": True}, callback=Tableau())
    print(result)
    print(result.x)

    # Ergebnis-Vektor des Simplex ausgeben
    for count in range(len(result.x)):
        print(nsimplify(result.x[count]))
    print(nsimplify(result.fun))

#    print(reduced_game)
#    for count in range(lines_game):
#        reducable_line = True
#        for count_2 in range(lines_game):
#            reducable_line = True
#            if count != count_2:
#                for count_3 in range(columns_game):
#                    if test_game[count][count_3] > test_game[count_2][count_3] and reducable_line:
#                        print(str(test_game[count][count_3]) + ' > ' + str(test_game[count_2][count_3]))
#                        reducable_line = False
#                if reducable_line:
#                    reduce.append([count+1, count_2+1])
#                    print(reducable_line)
#    print(reduced_game)

print('hallo test')



