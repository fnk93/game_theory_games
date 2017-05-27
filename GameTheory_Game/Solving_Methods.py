import numpy as np
from scipy import optimize
from sympy.solvers import solve
from sympy import nsimplify, symbols, Eq, Symbol
from copy import deepcopy
from random import randrange
from sympy.abc import u, v, x
import collections
import matplotlib.pyplot as plt


# Lösbar nach Nash:
# Alle Gleichgewichtspunkte sind vertauschbar
# Dominierte Gleichgewichtspunkte möglich


# mode = 0 -> normale Spiele
# mode = 1 -> Kampf der Geschlechter
# TODO: Auszahlungsdiagramm gemischte Strategien für mehr als 2 Stragien nutzbar machen.
def get_payoff_diagramm(payoff_matrix_1, payoff_matrix_2, mode=0):
    # payoff_player_1 = list()
    # payoff_player_2 = list()
    payoff_points = list()
    # functions = list()
    for lines in range(payoff_matrix_1.shape[0]):
        for columns in range(payoff_matrix_1.shape[1]):
            if [payoff_matrix_1[lines][columns], payoff_matrix_2[lines][columns]] not in payoff_points:
                payoff_points.append([payoff_matrix_1[lines][columns], payoff_matrix_2[lines][columns]])
                print(payoff_matrix_1[lines][columns], payoff_matrix_2[lines][columns])

    # Sort payoff_points

    for line_to_compare in range(0, len(payoff_points) - 1):
        min_distance = list()
        # print(line_to_compare)
        for compare_with in range(line_to_compare + 1, len(payoff_points)):
            # print(compare_with)
            min_distance.append(
                abs(payoff_points[line_to_compare][0] - payoff_points[compare_with][0]) + abs(
                    payoff_points[line_to_compare][1] - payoff_points[compare_with][1]))
        # print('minima: ', np.argmin(min_distance) + line_to_compare + 1)

        payoff_points[line_to_compare + 1], payoff_points[np.argmin(min_distance) + line_to_compare + 1] = \
            payoff_points[np.argmin(min_distance) + line_to_compare + 1], payoff_points[line_to_compare + 1].copy()

        # print(payoff_points)

        min_distance.clear()

    print(payoff_points)
    if mode == 1:
        # outline = ConvexHull(payoff_points, incremental=True)
        payoff_points = np.asarray(payoff_points)

        p = list()
        q = list()
        for lines in range(payoff_matrix_1.shape[0]):
            p.append(np.linspace(0, 1))
        for columns in range(payoff_matrix_2.shape[1]):
            q.append(np.linspace(0, 1))

        player_1_payoff = 0
        player_2_payoff = 0
        for lines in range(payoff_matrix_1.shape[0]):
            for columns in range(payoff_matrix_1.shape[1]):
                print(lines, columns)
                print(payoff_matrix_1[lines][columns])
                if lines > 0 and columns > 0:
                    player_1_payoff += payoff_matrix_1[lines][columns] * (1 - p[0]) * (1 - q[0])
                    player_2_payoff += payoff_matrix_2[lines][columns] * (1 - p[0]) * (1 - q[0])
                elif lines > 0 and columns == 0:
                    player_1_payoff += payoff_matrix_1[lines][columns] * (1 - p[0]) * q[columns]
                    player_2_payoff += payoff_matrix_2[lines][columns] * (1 - p[0]) * q[columns]
                elif columns > 0 and lines == 0:
                    player_1_payoff += payoff_matrix_1[lines][columns] * p[lines] * (1 - q[0])
                    player_2_payoff += payoff_matrix_2[lines][columns] * p[lines] * (1 - q[0])
                else:
                    player_1_payoff += payoff_matrix_1[lines][columns] * p[lines] * q[columns]
                    player_2_payoff += payoff_matrix_2[lines][columns] * p[lines] * q[columns]

        return payoff_points, [player_1_payoff, player_2_payoff]
    payoff_points.append(payoff_points[0])
    payoff_points = np.asarray(payoff_points)
    print('test', payoff_points)
    return [payoff_points]
    # return payoff_player_1, payoff_player_2, payoff_points


# TODO: Garantiepunkt dominiert in gemischten Strategien
# TODO: Gleichgewichtspunkt zurückgeben mit zugehörigem Strategiepaar(nash und Maximin)
# TODO: Nash-GGW + Wahrscheinlichkeiten + Payoff zusammenführen - done
# TODO: Spielwert-Berechnung bei unterschiedlichen Strategien (Gemischt, rein, Maximin) erledigt
# TODO: Maximin bei gemischten Strategien (=Nash-GGW!)
# TODO: Formulierung Lineares Programm ausgliedern aus Simplex
# TODO: Strategie- und Spielwert-Berechnung in Simplex
# TODO: Gleichgewichtspunkte bei Seitenzahlung / keiner Seitenzahlung -> Garantiepunkt nehmen
# TODO: max(u1 - u1G)*(u2 - u2G), NB: u1 >= u1G, u2 >= u2G
# TODO: Graphische Lösung gemischter Strategien


# Leerer Return bedeutet kein Nash-GGW
# Nash-GGW in reinen Strategien
def ggw(payoff_matrix_1, payoff_matrix_2, mode=0):
    optimal1 = np.zeros((payoff_matrix_1.shape[0], payoff_matrix_1.shape[1]))
    optimal2 = np.zeros((payoff_matrix_1.shape[0], payoff_matrix_1.shape[1]))
    dominated = list()
    result = list()
    if mode == 0:
        for column in range(payoff_matrix_1.shape[1]):
            max_val_1 = (np.argmax(payoff_matrix_1[:, column]))
            optimal1[max_val_1][column] += 1
            for line in range(payoff_matrix_1.shape[0]):
                if line != max_val_1 and payoff_matrix_1[line][column] == payoff_matrix_1[max_val_1][column]:
                    optimal1[line][column] += 1
            # print(max_val_1)

        for line in range(payoff_matrix_2.shape[0]):
            max_val_2 = (np.argmax(payoff_matrix_2[line]))
            optimal2[line][max_val_2] += 1
            for column in range(payoff_matrix_2.shape[1]):
                if column != max_val_2 and payoff_matrix_2[line][column] == payoff_matrix_2[line][max_val_2]:
                    optimal2[line][column] += 1
            # print(max_val_2)
        # print(optimal)
        optimal = optimal1 + optimal2
        prep = np.where(optimal == 2)
        for index in range(prep[0].shape[0]):
            result.append([prep[0][index], prep[1][index]])
        for ggws in range(len(result)):
            dominated_temp = False
            for lines in range(np.asarray(payoff_matrix_1).shape[0]):
                for columns in range(np.asarray(payoff_matrix_2).shape[1]):
                    if payoff_matrix_1[lines][columns] >= result[ggws][0] and \
                                    payoff_matrix_2[lines][columns] >= result[ggws][1]:
                        if payoff_matrix_1[lines][columns] != result[ggws][0] and \
                                        payoff_matrix_2[lines][columns] != result[ggws][1]:
                            dominated_temp = True
            dominated.append(dominated_temp)
    return result, dominated, optimal, optimal1, optimal2


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

    for ind in range(len(upper_values)):
        determination_intervalls.append([upper_values[ind], lower_values[ind]])

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
DD = np.asarray([[1, -1],
                 [-1, 1]])
print(determination_intervall(DD, DD*-1))

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
def bayes_strategy(payoff_matrix_1, payoff_matrix_2, ind, strategy):
    payoff_matrices = [payoff_matrix_1.transpose(), payoff_matrix_2]
    # print(payoff_matrices[player][strategy])
    bayes = (np.argmax(payoff_matrices[ind][strategy], 0))
    watched_strategy = payoff_matrices[ind][strategy]
    return bayes, watched_strategy


# Ergebnisse und Lösungswege als PDF formatieren
# TODO: evtl. in Game-Klasse übernehmen
# TODO: Auszahlungsdiagramme graphisch aufbereiten
# TODO: Simplex-Tableaus graphisch aufbereiten
# TODO: Parametrisierung welche Aufgaben gestellt wurden
def get_calculations_pdf(game, mode=0):
    matrix = game
    return matrix


# Ergebnisse und Lösungswege als LaTeX formatieren
# TODO: evtl. in Game-Klasse übernehmen
# TODO: Auszahlungsdiagramme graphisch aufbereiten
# TODO: Simplex-Tableaus graphisch aufbereiten
# TODO: Parametrisierung welche Aufgaben gestellt wurden
def get_calculations_latex(matrix1, matrix2, zerosum=False, bay1=0, bay2=0, mode=0):
    sol_tex = []
    sol_texpure = []
    sol_texmixed = []
    matrix1 = np.asarray(matrix1)
    matrix2 = np.asarray(matrix2)
    solution = ''
    context = {}
    maximins = solve_maximin_strategies(matrix1, matrix2)
    maximins_1 = []
    maximins_2 = []
    for element in maximins[0]:
        maximins_1.append(element+1)
    for element in maximins[1]:
        maximins_2.append(element+1)
    dets = determination_intervall(matrix1, matrix2)
    guarantee = get_guaranteed_payoff(matrix1, matrix2)
    played_strategy_1 = randrange(0, matrix1.shape[0])
    played_strategy_2 = randrange(0, matrix1.shape[1])
    bs_1 = bayes_strategy(matrix1, matrix2, 0, bay1)
    bayes_1 = bs_1[0]
    watch_1 = bs_1[1]
    bs_2 = bayes_strategy(matrix1, matrix2, 1, bay2)
    bayes_2 = bs_2[0]
    watch_2 = bs_2[1]
    equi = ggw(matrix1, matrix2, 0)
    low_values = get_lower_values(matrix1, matrix2)
    equi_points = []
    optimals = equi[2]
    optimals1 = equi[3]
    optimals2 = equi[4]
    determined = is_determined(matrix1, matrix2)
    for equis in equi[0]:
        equi_points.append([equis[0]+1,equis[1]+1])
    #equi_point = [equi[0][0][0]+1,equi[0][0][1]+1]
    solution += 'Bei der gegebenen Spielmatrix eines 2-Personen-Nullsummenspiels' + '\n'
    solution += str(matrix1) + '\n'
    sol_texpure.append(matrix1)
    temp_str = ''
    for lines in range(matrix1.shape[0]):
        for cols in range(matrix1.shape[1]):
            temp_str += str(matrix1[lines][cols]) + r'&'
        temp_str += r'\\'
    context['gamematrix'] = temp_str
    solution += 'ergeben sich folgende Kennzahlen: '+ '\n'
    solution += 'Maximin-Strategie(n) für Spieler 1: ' + str(maximins_1) + '\n'
    solution += 'Maximin-Strategie(n) für Spieler 2: ' + str(maximins_2) + '\n'
    sol_texpure.append(maximins_1)
    sol_texpure.append(maximins_2)
    context['maximin1'] = str(maximins_1)
    context['maximin2'] = str(maximins_2)
    solution += 'Indeterminiertheitsintervall: ' + str(dets[0]) + '\n'
    sol_texpure.append(dets[0])
    context['indet'] = str(dets[0])
    sol_texpure.append(determined)
    if determined:
        solution += 'Das Spiel ist somit determiniert.' + '\n'
        context['det'] = 'determiniert.'
    else:
        solution += 'Das Spiel ist somit indeterminiert.' + '\n'
        context['det'] = 'indeterminiert.'
    solution += 'Aus dem unteren Spielwert für Spieler 1: ' + str(low_values[0]) + '\n'
    context['lowval1'] = str(low_values[0])
    solution += 'und dem unteren Spielwert für Spieler 2: ' + str(low_values[1]) + '\n'
    context['lowval2'] = str(low_values[1])
    sol_texpure.append(low_values[0])
    sol_texpure.append(low_values[1])
    solution += 'ergibt sich der Garantiepunkt des Spiels in reinen Strategien: ' + str(guarantee[0]) + '\n'
    context['guar'] = str(guarantee[0])
    sol_texpure.append(guarantee[0])
    solution += 'Um die Bayes-Strategie zu ermitteln muss die maximale Auszahlung bei gegebener Gegnerstrategie betrachtet werden.' + '\n'
    solution += 'Für Spieler 1 müssen deshalb bei gegebener Strategie ' + str(played_strategy_2+1) + ' von Spieler 2 die Auszahlungen ' + str(watch_1) + ' betrachtet werden.' + '\n'
    solution += 'Hieraus ergibt sich die Bayes-Strategie : ' + str(bayes_1+1) + '\n'
    context['bay1'] = str(played_strategy_2+1)
    context['pay1'] = str(watch_1)
    context['baystrat1'] = str(bayes_1+1)
    sol_texpure.append(played_strategy_2+1)
    sol_texpure.append(watch_1)
    sol_texpure.append(bayes_1+1)
    solution += 'Für Spieler 2 müssen deshalb bei gegebener Strategie ' + str(
        played_strategy_1 + 1) + ' von Spieler 1 die Auszahlungen ' + str(watch_2) + ' betrachtet werden.' + '\n'
    solution += 'Hieraus ergibt sich die Bayes-Strategie : ' + str(bayes_2+1) + '\n'
    context['bay2'] = str(played_strategy_1 + 1)
    context['pay2'] = str(watch_2)
    context['baystrat2'] = str(bayes_2 + 1)
    sol_texpure.append(played_strategy_1 + 1)
    sol_texpure.append(watch_2)
    sol_texpure.append(bayes_2 + 1)
    solution += 'Das Erfüllen der Optimalitätsbedingung der Strategiekombinationen über beide Spieler aufsummiert sieht wie folgt aus:'+ '\n'
    solution += str(optimals) + '\n'
    temp_str = ''
    for lines in range(optimals.shape[0]):
        for cols in range(optimals.shape[1]):
            temp_str += str(optimals[lines][cols]) + '&'
        temp_str += r'\\'
    context['ggwmatr'] = temp_str
    sol_texpure.append(optimals)
    solution += 'Wobei sich für Spieler 1 folgende Verteilung der Erfüllung der Optimalitätsbedingung' + '\n'
    solution += str(optimals1) + '\n'
    sol_texpure.append(optimals1)
    temp_str = ''
    for lines in range(optimals1.shape[0]):
        for cols in range(optimals1.shape[1]):
            temp_str += str(optimals1[lines][cols]) + '&'
        temp_str += r'\\'
    context['ggwmatr1'] = temp_str
    solution += 'und sich für Spieler 2 folgende Verteilung der Erfüllung der Optimalitätsbedingung ergab' + '\n'
    solution += str(optimals2) + '\n'
    temp_str = ''
    for lines in range(optimals2.shape[0]):
        for cols in range(optimals2.shape[1]):
            temp_str += str(optimals2[lines][cols]) + '&'
        temp_str += r'\\'
    context['ggwmatr2'] = temp_str
    sol_texpure.append(optimals2)
    if len(equi_points) > 0:
        solution += 'Jede Strategiekombination, die sowohl für Spieler 1, als auch für Spieler 2 die Optimalitätsbedingung erfüllt ist Gleichgewichtspunkt des Spiels in reinen Strategien' + '\n'
        solution += 'Gleichgewichtspunkt(e): '+ str(equi_points) + ' mit zugehöriger Auszahlung für Spieler 1: ' + str(matrix1[equi[0][0][0]][equi[0][0][1]]) + '\n' + 'und Auszahlung für Spieler 2: ' + str(matrix2[equi[0][0][0]][equi[0][0][1]])
        sol_texpure.append(equi_points)
        sol_texpure.append(matrix1[equi[0][0][0]][equi[0][0][1]])
        sol_texpure.append(matrix2[equi[0][0][0]][equi[0][0][1]])
        context['puresolve'] = 'Jede Strategiekombination, die sowohl für Spieler 1, als auch für Spieler 2 die Optimalitätsbedingung erfüllt ist Gleichgewichtspunkt des Spiels in reinen Strategien\\Gleichgewichtspunkt(e): '+ str(equi_points) + ' mit zugehöriger Auszahlung für Spieler 1: ' + str(matrix1[equi[0][0][0]][equi[0][0][1]]) + '\n' + 'und Auszahlung für Spieler 2: ' + str(matrix2[equi[0][0][0]][equi[0][0][1]]) + r'\\'
    else:
        solution += 'Da keine Strategiekombination sowohl für Spieler 1, als auch für Spieler 2 die Optimalitätsbedingung erfüllt existiert kein Gleichgewichtspunkt in reinen Strategien'
        sol_texpure.append([])
        sol_texpure.append([])
        sol_texpure.append([])
        context['puresolve'] = 'Da keine Strategiekombination sowohl für Spieler 1, als auch für Spieler 2 die Optimalitätsbedingung erfüllt existiert kein Gleichgewichtspunkt in reinen Strategien\\'
    context['solvemixed'] = ""
    if mode > 0:
        if not determined:
            context['solvemixed'] = ''
            if zerosum:
                simplex = use_simplex(matrix1, matrix2)
                if simplex[2][0].all() != matrix1.all():
                    solution += 'Aufgrund der Beschränkungen des Simplex-Algorithmus muss zunächst die Auszahlungsmatrix des betrachteten Zwei-Personen-Nullsummenspiels absolut positiv werden.' + '\n'
                    solution += 'Die zu lösende Matrix sieht nun folgendermaßen aus: ' + '\n'
                    solution += str(simplex[2][0]) + '\n'
                    sol_texmixed.append(simplex[2][0])
                    context['solvemixed'] += r'Aufgrund der Beschränkungen des Simplex-Algorithmus muss zunächst die Auszahlungsmatrix des betrachteten Zwei-Personen-Nullsummenspiels absolut positiv werden.\\Die zu lösende Matrix sieht nun folgendermaßen aus:\\'
                    context['solvemixed'] += r'\begin{gather*}\begin{pmatrix*}'
                    temp_str = ''
                    for lines in range(simplex[2][0].shape[0]):
                        for cols in range(simplex[2][0].shape[1]):
                            temp_str += str(simplex[2][0][lines][cols]) + '&'
                        temp_str += r'\\'
                    context['solvemixed'] += temp_str + '\end{pmatrix*}\end{gather*}'
                else:
                    solution += 'Das folgende Spiel soll nun mithilfe des Simplex-Algorithmus gelöst werden:' + '\n'
                    solution += str(matrix1)
                    sol_texmixed.append(matrix1)
                    context['solvemixed'] += r'Das folgende Spiel soll nun mithilfe des Simplex-Algorithmus gelöst werden:\\'
                    temp_str = ''
                    for lines in range(matrix1.shape[0]):
                        for cols in range(matrix1.shape[1]):
                            temp_str += str(matrix1[lines][cols]) + '&'
                        temp_str += r'\\'
                    context['solvemixed'] += temp_str + r'\\'
                solution += 'Die Lösungsschritte des Simplexalgorithmus für Spieler 2 sehen nun wie folgt aus: ' + '\n'
                context['solvemixed'] += r'Die Lösungsschritte des Simplexalgorithmus für Spieler 2 sehen nun wie folgt aus:\\ ' + r' \begin{gather*}'
                #context['solvemixed'] += r'\begin{gather*}'
                for step in simplex[1][1:][0]:
                    temp_arr = np.asarray(format_solution(step['tableau']))
                    solution += str(temp_arr) + '\n'
                    sol_texmixed.append(temp_arr)
                    temp_str = ''
                    for lines in range(temp_arr.shape[0]):
                        for cols in range(temp_arr.shape[1]):
                            temp_str += str(temp_arr[lines][cols]) + '&'
                        temp_str += r'\\'
                    context['solvemixed'] += r'\begin{pmatrix*}\\' + temp_str + r'\end{pmatrix*}\\'
                    if str(step['pivot']) != '(nan, nan)':
                        solution += 'Pivot: ' + str(step['pivot']) + '\n'
                        sol_texmixed.append(step['pivot'])
                        context['solvemixed'] += 'Pivot: ' + str(step['pivot']) + r'\\'
                    else:
                        sol_texmixed.append([])
                context['solvemixed'] += r'\end{gather*}\\'
                added_value = np.amax(simplex[2][0]-matrix1)
                solution += 'Da nicht der Spielwert maximiert wurde, sondern 1/G minimiert wurde und eine Konstante ' + str(added_value) + ' zur Matrix addiert wurde, muss man die Konstante wieder vom Ergebnis subtrahieren und den Kehrbruch verwenden.' + '\n'
                sol_texmixed.append(added_value)
                context['solvemixed'] += r'Da nicht der Spielwert maximiert wurde, sondern 1/G minimiert wurde und eine Konstante ' + str(added_value) + r' zur Matrix addiert wurde, muss man die Konstante wieder vom Ergebnis subtrahieren und den Kehrbruch verwenden.\\'
                game_value_1 = nsimplify((1/simplex[1][0]['fun']) + added_value, tolerance=0.0001, rational=True)
                strategies = []
                for strategy in simplex[1][0]['x']:
                    strategies.append(nsimplify(abs(((1/simplex[1][0]['fun'])*strategy)), tolerance=0.0001, rational=True))
                solution += 'Hieraus ergibt sich der tatsächliche Spielwert für Spieler 2: ' + str(game_value_1) + '\n'
                context['solvemixed'] += 'Hieraus ergibt sich der tatsächliche Spielwert für Spieler 2: ' + str(game_value_1) + r'\\'
                sol_texmixed.append(game_value_1)
                solution += 'Und die optimale Strategienkombination für Spieler 2: ' + str(strategies) + '\n'
                context['solvemixed'] += 'Und die optimale Strategienkombination für Spieler 2: ' + str(strategies) + r'\\'
                sol_texmixed.append(strategies)
                solution += 'Da ein Zwei-Personen-Nullsummenspiel betrachtet wurde ergibt sich der Spielwert für Spieler 1: ' + str(game_value_1*-1) + '\n'
                context['solvemixed'] += 'Da ein Zwei-Personen-Nullsummenspiel betrachtet wurde ergibt sich der Spielwert für Spieler 1: ' + str(game_value_1*-1) + r'\\'
                strategies_1 = []
                sol_texmixed.append(game_value_1*-1)
                possible_choices = -1 - matrix1.shape[0]
                for elements in simplex[1][1][-1]['tableau'][-1][possible_choices:-1]:
                    strategies_1.append(nsimplify(elements*(1/simplex[1][1][-1]['tableau'][-1][-1]), tolerance=0.0001, rational=True))
                solution += 'Die Dualität des Problems erlaubt es, die optimale Strategienkombination für Spieler 1 direkt aus der Zielfunktionszeile abzulesen: ' + str(strategies_1) + '\n'
                sol_texmixed.append(strategies_1)
                context['solvemixed'] += 'Die Dualität des Problems erlaubt es, die optimale Strategienkombination für Spieler 1 direkt aus der Zielfunktionszeile abzulesen: ' + str(strategies_1) + r'\\'
                solution += '\n\n\n'
                reduced = reduce_matrix(matrix1, matrix2)
                boole = reduced[2]
                #if boole:
                #    solution += 'Zunächst müssen die überflüssigen Zeilen und Spalten der Ausgangsmatrix entfernt werden.' + '\n'
                #    solution += 'Hierdurch ergibt sich folgende Matrix für Spieler 1:' + '\n'
                #    solution += str(reduced[0]) + '\n'

                try:
                    nggw = solve_using_nggw(matrix1, matrix2)
                    solution_1 = nggw[0]
                    lgs1 = solution_1[2]
                    solution_2 = nggw[1]
                    lgs2 = solution_2[2]
                    correct_solution = True
                    for key in solution_1[1]:
                        if solution_1[0][key] < 0 and key != solution_1[1][-1]:
                            correct_solution = False
                            solution += 'Key-Fehler: ' + str(solution_1[0][key]) + ' ' + str(key)+ '\n'
                    for key in solution_2[1]:
                        if solution_2[0][key] < 0 and key != solution_2[1][-1]:
                            correct_solution = False
                            solution += 'Key-Fehler: ' + str(solution_2[0][key]) + ' ' + str(key) + '\n'
                    if correct_solution:
                        solution += 'Selbiges Problem lässt sich auch durch die Aufstellung eines LGS nach den Bedingungen für ein Nash-Gleichgewicht lösen: ' + '\n'
                        context['solvemixed'] += r'Selbiges Problem lässt sich auch durch die Aufstellung eines LGS nach den Bedingungen für ein Nash-Gleichgewicht lösen:\\'
                        if len(solution_1[0]) > 0:
                            solution += 'Das lineare Gleichungssystem für Spieler 1 lautet: ' + '\n'
                            context['solvemixed'] += r'Das lineare Gleichungssystem für Spieler 1 lautet:\\'
                            temp = []
                            for equation in lgs1:
                                solution += str((equation))
                                solution += '\n'
                                temp.append(equation)
                                context['solvemixed'] += str(equation) + r'\\'
                            sol_texmixed.append(temp)
                            solution += 'Nach Auflösen des LGS ergeben sich folgene Werte für die optimale Strategienkombination: ' + '\n'
                            context['solvemixed'] += r'Nach Auflösen des LGS ergeben sich folgene Werte für die optimale Strategienkombination:\\'
                            temp = []
                            for key in solution_1[1]:
                                solution += str(key) + ':' + str(nsimplify(solution_1[0][key], tolerance=0.0001, rational=True)) + '\n'
                                val = nsimplify(solution_1[0][key], tolerance=0.0001, rational=True)
                                temp.append([key, val])
                                context['solvemixed'] += str(key) + ':' + str(val) + r'\\'
                            sol_texmixed.append(temp)
                        if len(solution_2[0]) > 0:
                            solution += 'Das lineare Gleichungssystem für Spieler 2 lautet: ' + '\n'
                            context['solvemixed'] += r'Das lineare Gleichungssystem für Spieler 2 lautet:\\'
                            temp = []
                            for equation in lgs2:
                                solution += str(equation)
                                temp.append(equation)
                                solution += '\n'
                                context['solvemixed'] += str(equation) + r'\\'
                            sol_texmixed.append(temp)
                            solution += 'Nach Auflösen des LGS ergeben sich folgene Werte für die optimale Strategienkombination: ' + '\n'
                            context[
                                'solvemixed'] += r'Nach Auflösen des LGS ergeben sich folgene Werte für die optimale Strategienkombination:\\'
                            temp = []
                            for key in solution_2[1]:
                                solution += str(key) + ':' + str(nsimplify(solution_2[0][key], tolerance=0.0001, rational=True)) + '\n'
                                val = nsimplify(solution_2[0][key], tolerance=0.0001, rational=True)
                                temp.append([key, val])
                                context['solvemixed'] += str(key) + ':' + str(val) + r'\\'
                            sol_texmixed.append(temp)
                except:
                    pass
            else:
                try:
                    nggw = solve_using_nggw(matrix1, matrix2)
                    solution_1 = nggw[0]
                    lgs1 = solution_1[2]
                    solution_2 = nggw[1]
                    lgs2 = solution_2[2]
                    correct_solution = True
                    for key in solution_1[1]:
                        if solution_1[0][key] < 0 and key != solution_1[1][-1]:
                            correct_solution = False
                            solution += 'Key-Fehler: ' + str(solution_1[0][key]) + ' ' + str(key)+ '\n'
                    for key in solution_2[1]:
                        if solution_2[0][key] < 0 and key != solution_2[1][-1]:
                            correct_solution = False
                            solution += 'Key-Fehler: ' + str(solution_2[0][key]) + ' ' + str(key) + '\n'
                    if correct_solution:
                        solution += 'Selbiges Problem lässt sich auch durch die Aufstellung eines LGS nach den Bedingungen für ein Nash-Gleichgewicht lösen: ' + '\n'
                        context['solvemixed'] += r'Selbiges Problem lässt sich auch durch die Aufstellung eines LGS nach den Bedingungen für ein Nash-Gleichgewicht lösen:\\'
                        if len(solution_1[0]) > 0:
                            solution += 'Das lineare Gleichungssystem für Spieler 1 lautet: ' + '\n'
                            context['solvemixed'] += r'Das lineare Gleichungssystem für Spieler 1 lautet:\\'
                            temp = []
                            for equation in lgs1:
                                solution += str((equation))
                                solution += '\n'
                                temp.append(equation)
                                context['solvemixed'] += str(equation) + r'\\'
                            sol_texmixed.append(temp)
                            solution += 'Nach Auflösen des LGS ergeben sich folgene Werte für die optimale Strategienkombination: ' + '\n'
                            context['solvemixed'] += r'Nach Auflösen des LGS ergeben sich folgene Werte für die optimale Strategienkombination:\\'
                            temp = []
                            for key in solution_1[1]:
                                solution += str(key) + ':' + str(nsimplify(solution_1[0][key], tolerance=0.0001, rational=True)) + '\n'
                                val = nsimplify(solution_1[0][key], tolerance=0.0001, rational=True)
                                temp.append([key, val])
                                context['solvemixed'] += str(key) + ':' + str(val) + r'\\'
                            sol_texmixed.append(temp)
                        if len(solution_2[0]) > 0:
                            solution += 'Das lineare Gleichungssystem für Spieler 2 lautet: ' + '\n'
                            context['solvemixed'] += r'Das lineare Gleichungssystem für Spieler 2 lautet:\\'
                            temp = []
                            for equation in lgs2:
                                solution += str(equation)
                                temp.append(equation)
                                solution += '\n'
                                context['solvemixed'] += str(equation) + r'\\'
                            sol_texmixed.append(temp)
                            solution += 'Nach Auflösen des LGS ergeben sich folgene Werte für die optimale Strategienkombination: ' + '\n'
                            context[
                                'solvemixed'] += r'Nach Auflösen des LGS ergeben sich folgene Werte für die optimale Strategienkombination:\\'
                            temp = []
                            for key in solution_2[1]:
                                solution += str(key) + ':' + str(nsimplify(solution_2[0][key], tolerance=0.0001, rational=True)) + '\n'
                                val = nsimplify(solution_2[0][key], tolerance=0.0001, rational=True)
                                temp.append([key, val])
                                context['solvemixed'] += str(key) + ':' + str(val) + r'\\'
                            sol_texmixed.append(temp)
                except:
                    pass

    sol_tex = [sol_texpure, sol_texmixed]
    return solution, sol_tex, context


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
                # reducable_line = True
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
                # reducable_column = True
                added = False
                for count_2 in range(dimensions[1]):
                    reducable_column = True
                    if count != count_2:
                        for count_3 in range(dimensions[0]):
                            if reduced_matrix_2[count_3][count] > reduced_matrix_2[count_3][count_2]:
                                if reducable_column:
                                    reducable_column = False
                        if reducable_column and not added:
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
    # inner_copy_1 = np.asarray(payoff_matrix_1)  # type: np.ndarray
    # inner_copy_2 = np.asarray(payoff_matrix_2)  # type: np.ndarray
    if mode == 0:
        payoff = get_lower_values(payoff_matrix_1, payoff_matrix_2)
        for lines in range(np.asarray(payoff_matrix_1).shape[0]):
            for columns in range(np.asarray(payoff_matrix_2).shape[1]):
                if payoff_matrix_1[lines][columns] >= payoff[0] and payoff_matrix_2[lines][columns] >= payoff[1]:
                    if payoff_matrix_1[lines][columns] != payoff[0] and payoff_matrix_2[lines][columns] != payoff[1]:
                        dominated = True
    elif mode == 1:
        result = solve_using_nggw(payoff_matrix_1, payoff_matrix_2)
        # print(result)
        for players in range(len(result)):
            # print(result[players][1][-1])
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
    # print('Matrizen: ', simplex_games)
    simplex_1_solution = use_simplex_player1(simplex_games[1])
    simplex_2_solution = use_simplex_player2(simplex_games[0])

    return [simplex_1_solution, simplex_2_solution, simplex_games]


# Simplex-Verfahren für Spieler 1 anwenden
# TODO: Formatierung des Lösungswegs direkt hier machen
def use_simplex_player2(simplex_game_1):
    c = list()
    game_bounds = list()
    a = list()
    b = list()
    for lines in range(np.asarray(simplex_game_1).shape[0]):
        temp = list()
        for columns in range(np.asarray(simplex_game_1.shape[1])):
            temp.append(simplex_game_1[lines][columns])
        a.append(temp)
        b.append(1)

    for columns in range(np.asarray(simplex_game_1.shape[1])):
        c.append(-1)
        game_bounds.append((0, None))

    # xk_arr1 = list()
    # kwargs_arr1 = list()
    solve_report_1 = SolvingSteps()
    simplex_sol = optimize.linprog(c, a, b, callback=solve_report_1.save_values)
    simplex_steps = solve_report_1.get_array_kwargs()
    simplex_steps_2 = format_solution(solve_report_1.get_array_xk())

    # print('Player 2: ', c, A, b)

    return simplex_sol, simplex_steps, simplex_steps_2


# Simplex-Verfahren für Spieler 2 anwenden
# TODO: Formatierung des Lösungswegs direkt hier machen
def use_simplex_player1(simplex_game_2):
    c = list()
    game_bounds = list()
    a = list()
    b = list()
    for lines in range(np.asarray(simplex_game_2).shape[0]):
        c.append(1)
        game_bounds.append((0, None))

    for columns in range(np.asarray(simplex_game_2.shape[1])):
        b.append(-1)
        temp = list()
        for lines in range(np.asarray(simplex_game_2).shape[0]):
            temp.append(simplex_game_2[lines][columns])
        a.append(temp)

    # xk_arr2 = list()
    # kwargs_arr2 = list()
    solve_report2 = SolvingSteps()
    simplex_sol = optimize.linprog(c, a, b, callback=solve_report2.save_values)
    simplex_steps = solve_report2.get_array_kwargs()
    simplex_steps_2 = format_solution(solve_report2.get_array_xk())

    # print('Player 1: ', c, A, b)

    return simplex_sol, simplex_steps, simplex_steps_2


# Lösung mit Bedingungen für NGGW
# Gemischte Maximin-Strategien der Spieler
def solve_using_nggw(payoff_matrix_1, payoff_matrix_2):
    # Gemischte Strategien p für Spieler 1 und Spielwert w für Spieler 2
    payoff_matrice = reduce_matrix(payoff_matrix_1, payoff_matrix_2)
    payoff_matrix_1 = payoff_matrice[0]
    payoff_matrix_2 = payoff_matrice[1]
    # Variablen des LGS deklarieren
    #p = symbols('p:' + str(payoff_matrix_2.shape[0]), nonnegative=True)
    #v = Symbol("New symbol", positive=True)
    #print(v.assumptions0)
    p = []
    w = Symbol('w', real=True)
    for var in range(payoff_matrix_2.shape[0]):
        p.append(Symbol('p'+str(var), nonnegative=True))
    # Lösungssystem erstellen und mit Gleichungen füllen
    # Variablensystem erstellen und füllen
    u = list()
    for column in range(payoff_matrix_2.shape[1]):
        temp = 0
        for line in range(payoff_matrix_2.shape[0]):
            temp += payoff_matrix_2[line][column] * p[line]
        u.append(temp-w)
    temp_2 = 0
    symbol = list()
    for decisions in range(len(p)):
        temp_2 += 1 * p[decisions]
        symbol.append(p[decisions])
    u.append(temp_2-1)
    symbol.append(w)
    # LGS lösen und speichern für Rückgabe
    solution_1 = solve(u, force=True)
    solution = list()

    # Gemischte Strategien q für Spieler 2 und Spielwert w für Spieler 1

    # Variablen des LGS deklarieren
    #q = symbols('q:' + str(payoff_matrix_1.shape[1]), positive=True, nonnegative=True)
    w2 = Symbol('w', real=True)
    q = []
    # Lösungssystem erstellen und mit Gleichungen füllen
    # Variablensystem erstellen und füllen
    u2 = list()
    for var in range(payoff_matrix_1.shape[1]):
        name = 'q' + str(var)
        temp_sym = Symbol(name, nonnegative=True)
        q.append(temp_sym)
    for line in range(payoff_matrix_1.shape[0]):
        temp = 0
        for column in range(payoff_matrix_1.shape[1]):
            temp += payoff_matrix_1[line][column] * q[column]
        temp -= w2
        u2.append(temp)
    temp_2 = 0
    symbol2 = list()
    for decisions in range(len(q)):
        temp_2 += 1 * q[decisions]
        symbol2.append(q[decisions])
    u2.append(temp_2-1)
    symbol2.append(w2)
    # LGS lösen und speichern für Rückgabe
    solution_2 = solve(u2, check=False, force=True)
    # print(u, u2)
    # print(solution_1, solution_2)
    # print(symbol, symbol2)
    # print(solution_1[symbol[-1]], solution_2[symbol2[-1]])
    # print(solution_1[symbol[-1]].copy())
    if len(solution_1) > 0 and len(solution_2) > 0:
        solution_1[symbol[-1]], solution_2[symbol2[-1]] = solution_2[symbol2[-1]], solution_1[symbol[-1]]
        symbol[-1], symbol2[-1] = symbol2[-1], symbol[-1]

    solution.append([solution_1, symbol, u])
    solution.append([solution_2, symbol2, u2])

    # print(solution)

    return solution


def format_solution(solution_array):
    # print('Formatting: ')
    solution = list()
    # print(np.asarray(solution_array))
    # print('Formatting:')
    # print(np.asarray(solution_array))
    for line in range(np.asarray(solution_array).shape[0]):
        temp = list()
        for column in range(np.asarray(solution_array).shape[1]):
            temp.append(nsimplify(solution_array[line][column], tolerance=0.0001, rational=True))
            # print(nsimplify(solution_array[line][column], tolerance=0.0001, rational=True))
        if np.amin(temp) != 0 or np.amax(temp) != 0:
            solution.append(temp)
    # print(solution)

    return solution


# Callable Methode um Zwischenschritte des Simplex abzufangen
class SolvingSteps:
    def __init__(self):
        self.__array_xk = []
        self.__array_kwargs = []
        # print('Initialisiert mit:')
        # print('xk:')
        # print(self.__array_xk)
        # print('kwargs:')
        # print(self.__array_kwargs)

    def save_values(self, xk, **kwargs):
        # print('Speichern:')
        # print(xk)
        # print(self.__array_xk)
        # for i in range(len(self.__array_xk)):
        #     print(self.__array_xk[i])
        # self.array_xk = xk
        # print('xk:')
        # print(np.asarray(xk))
        # print('array xk:')
        # print(self.get_array_xk())
        temp_values = {}
        for key, value in kwargs.items():
            temp_values[key] = deepcopy(value)
            # print('Key-Value-Paar:')
            # print(key, value)
        # print(temp_values['tableau'])
        self.__array_kwargs.append(deepcopy(temp_values))
        # print('nach kwargs:')
        # print(self.__array_xk)
        # print(xk)
        self.__array_xk.append(deepcopy(xk))
        # print('added:')
        # print(xk)
        # print(self.__array_xk)

    # Funktion um Dictionaries auszugeben
    def get_array_kwargs(self):
        return self.__array_kwargs

    # Funktion um Parameter des Simplex abzufragen
    def get_array_xk(self):
        return self.__array_xk


