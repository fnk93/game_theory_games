import numpy as np
from scipy import optimize
from sympy.solvers import solve
from sympy import nsimplify, symbols, Symbol
from copy import deepcopy
from itertools import chain, combinations, product


def powerset(n):
    """Erzeugt einen Iterator, um die möglichen Supports in gemischten Strategien abzubilden

    :param n: Anzahl der unterschiedlichen Werte, die kombiniert werden sollen
    :returns: Einen Iterator, der alle Kombinationen von bis zu n Elementen der Werte von 0 bis n enthält
    :rtype: chain
    """
    return chain.from_iterable(combinations(range(n), r) for r in range(n + 1))


# mode = 0 -> normale Spiele
# mode = 1 -> Kampf der Geschlechter
# TODO: Auszahlungsdiagramm gemischte Strategien für mehr als 2 Stragien nutzbar machen.
def get_payoff_diagramm(payoff_matrix_1, payoff_matrix_2=np.array([]), mode=0):
    """Erzeugt ein Auszahlungsdiagramm in reinen oder gemischten Strategien

    :param payoff_matrix_1: Die Auszahlungsmatrix von Spieler 1
    :type payoff_matrix_1: ndarray
    :param payoff_matrix_2: Die Auszahlungsmatrix von Spieler 2 (default: np.array([]))
    :type payoff_matrix_2: ndarray
    :param mode: Soll in reinen (mode=0) oder gemischten (mode=1) Strategien gearbeitet werden (default: 0)
    :type mode: int
    :return: Eine Liste der Auszahlungspunkte
    :rtype: list
    """
    if not payoff_matrix_2.size:
        payoff_matrix_2 = payoff_matrix_1*-1
    # payoff_player_1 = list()
    # payoff_player_2 = list()
    payoff_points = list()
    # functions = list()
    for lines in range(payoff_matrix_1.shape[0]):
        for columns in range(payoff_matrix_1.shape[1]):
            if [payoff_matrix_1[lines][columns], payoff_matrix_2[lines][columns]] not in payoff_points:
                payoff_points.append([payoff_matrix_1[lines][columns], payoff_matrix_2[lines][columns]])
                #print(payoff_matrix_1[lines][columns], payoff_matrix_2[lines][columns])

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

        del min_distance[:]

    #print(payoff_points)
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
                #print(lines, columns)
                #print(payoff_matrix_1[lines][columns])
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
    #print('test', payoff_points)
    return [payoff_points]
    # return payoff_player_1, payoff_player_2, payoff_points


# TODO: Garantiepunkt dominiert in gemischten Strategien
# TODO: Gleichgewichtspunkt zurückgeben mit zugehörigem Strategiepaar(nash und Maximin)
# TODO: Nash-GGW + Wahrscheinlichkeiten + Payoff zusammenführen - done
# TODO: Spielwert-Berechnung bei unterschiedlichen Strategien (Gemischt, rein, Maximin) erledigt
# TODO: Maximin bei gemischten Strategien (=Nash-GGW!)
# TODO: Formulierung Lineares Programm ausgliedern aus Simplex
# TODO: Strategie- und Spielwert-Berechnung in Simplex
# TODO: Graphische Lösung gemischter Strategien


# Leerer Return bedeutet kein Nash-GGW
# Nash-GGW in reinen Strategien
def ggw(payoff_matrix_1, payoff_matrix_2=np.array([])):
    """Bestimmt die Nash-Gleichgewichte in reinen Strategien

    :param payoff_matrix_1: Die Auszahlungsmatrix von Spieler 1
    :type payoff_matrix_1: ndarray
    :param payoff_matrix_2: Die Auszahlungsmatrix von Spieler 2 (default: np.array([]))
    :type payoff_matrix_2: ndarray
    :return: Eine Liste, die die Strategie-Kombinationen, eine Stabilitäts-Aussage und den Rechenweg für jedes Nash-\
    Gleichgewicht beinhaltet
    :rtype: list
    """
    if not payoff_matrix_2.size:
        payoff_matrix_2 = payoff_matrix_1*-1
    optimal1 = np.zeros(payoff_matrix_1.shape)
    optimal2 = np.zeros(payoff_matrix_1.shape)
    #optimal = 0
    dominated = list()
    result = list()

    for column in range(payoff_matrix_1.shape[1]):
        opt = bayes_strategy(payoff_matrix_1.T, column)[0]
        for bay in opt:
            optimal1[bay[0]][column] += 1
    #for column in range(payoff_matrix_1.shape[1]):
    #    max_val_1 = (np.argmax(payoff_matrix_1[:, column]))
    #    optimal1[max_val_1][column] += 1
    #    for line in range(payoff_matrix_1.shape[0]):
    #        if line != max_val_1 and payoff_matrix_1[line][column] == payoff_matrix_1[max_val_1][column]:
    #            optimal1[line][column] += 1
        # print(max_val_1)
    for line in range(payoff_matrix_2.shape[0]):
        opt = bayes_strategy(payoff_matrix_2, line)[0]
        for bay in opt:
            optimal2[line][bay[0]] += 1
    #for line in range(payoff_matrix_2.shape[0]):
    #    max_val_2 = (np.argmax(payoff_matrix_2[line]))
    #    optimal2[line][max_val_2] += 1
    #    for column in range(payoff_matrix_2.shape[1]):
    #        if column != max_val_2 and payoff_matrix_2[line][column] == payoff_matrix_2[line][max_val_2]:
    #            optimal2[line][column] += 1
        # print(max_val_2)
    # print(optimal)
    optimal = optimal1 + optimal2
    prep = np.where(optimal == 2)
    #result = [[prep[0][index], prep[1][index]] for index in range(prep[0].shape[0])]]
    result = []
    for index in range(prep[0].shape[0]):
        result.append([prep[0][index], prep[1][index]])
    #print(result, len(result), np.asarray(result[0]).size)
    for ggws in range(len(result)):
        dominated_temp = False
        for lines in range(np.asarray(payoff_matrix_1).shape[0]):
            for columns in range(np.asarray(payoff_matrix_2).shape[1]):
                #print(lines, columns, ggws)
                if payoff_matrix_1[lines][columns] >= result[ggws][0] and \
                                payoff_matrix_2[lines][columns] >= result[ggws][1]:
                    if payoff_matrix_1[lines][columns] != result[ggws][0] and \
                                    payoff_matrix_2[lines][columns] != result[ggws][1]:
                        dominated_temp = True
        dominated.append(dominated_temp)
    return result, dominated, optimal, optimal1, optimal2


# Prüft ob für jeden Spieler unterer Spielwert dem oberen entspricht
def is_determined(payoff_matrix_1):
    """Prüft ob das Spiel determiniert ist

    :param payoff_matrix_1: Die Auszahlungsmatrix von Spieler 1
    :type payoff_matrix_1: ndarray
    :returns: Den Wahrheitswert, ob ein Spiel determiniert oder nicht ist
    :rtype: boolean
    """
    det_intervalls = determination_intervall(payoff_matrix_1)
    #print(det_intervalls)
    if det_intervalls[0] != det_intervalls[1]:
        return False

    return True


# Determiniertheitsintervall für beide Spieler berechnen
def determination_intervall(payoff_matrix_1):
    """Bestimmt das Indeterminiertheitsintervall eines Spiels

    :param payoff_matrix_1: Die Auszahlungsmatrix von Spieler 1
    :type payoff_matrix_1: ndarray
    :returns: Das Indeterminiertheitsintervall als Liste
    :rtype: list
    """
    upper_values = get_upper_values(payoff_matrix_1)
    lower_values = get_lower_values(payoff_matrix_1)
    determination_intervalls = [lower_values[0], upper_values]

    return determination_intervalls


# Obere Spielwerte für beide Spieler in reinen Strategien ermitteln
def get_upper_values(payoff_matrix_1):
    """Bestimmt den oberen Spielwert eines Zwei-Personennullsummenspiels

    :param payoff_matrix_1: Die Auszahlungsmatrix von Spieler 1
    :type payoff_matrix_1: ndarray
    :returns: Wert des oberen Spielwerts
    :rtype: int
    """
    upper_values = min([max(payoff_matrix_1[:, col]) for col in range(payoff_matrix_1.shape[1])])
    #temp_values = list()
    #for column in range(payoff_matrix_1.shape[1]):
    #    temp_values.append(max(payoff_matrix_1[:, column]))
    #upper_values = [deepcopy(min(temp_values))]
    #del temp_values[:]

    return upper_values


# Untere Spielwerte für beide Spieler in reinen Strategien ermitteln
# Spieler 1: Minimum der einzelnen Zeilen, davon das Maximum
# Spieler 2: Minimum der einzelnen Spalten, davon das Maximum
def get_lower_values(payoff_matrix_1, payoff_matrix_2=np.array([])):
    """Bestimmt die unteren Spielwerte der Spieler in einem Zwei-Personenspiel

    :param payoff_matrix_1: Die Auszahlungsmatrix von Spieler 1
    :type payoff_matrix_1: ndarray
    :param payoff_matrix_2: Die Auszahlungsmatrix von Spieler 2 (default: np.array([]))
    :type payoff_matrix_2: ndarray
    :returns: Liste der unteren Spielwerte der beiden Spieler
    :rtype: list
    """
    if not payoff_matrix_2.size:
        payoff_matrix_2 = (payoff_matrix_1*-1)
    payoff_matrix_2 = payoff_matrix_2.T
    #temp_values = list()
    #temp_values2 = list()
    #for line in range(payoff_matrix_1.shape[0]):
    #    temp_values.append(min(payoff_matrix_1[line]))
    #    temp_values2.append(min(payoff_matrix_2[line]))
    #lower_values = [deepcopy(max(temp_values)), deepcopy(max(temp_values2))]
    lower_values = [max([min(payoff_matrix_1[line]) for line in range(payoff_matrix_1.shape[0])]),
                    max([min(payoff_matrix_2[line]) for line in range(payoff_matrix_2.shape[0])])]
    #del temp_values[:]
    #del temp_values2[:]

    #for columns in range(payoff_matrix_2.shape[1]):
    #    temp_values.append(min(payoff_matrix_2[:, columns]))
    #lower_values.append(deepcopy(max(temp_values)))
    #del temp_values[:]

    return lower_values


# Maximin-Strategien der Spieler
# Sollte nur bei determinierten Spielen angewendet werden
def solve_maximin_strategies(payoff_matrix_1, payoff_matrix_2=np.array([])):
    """Erzeugt eine Liste aller Maximin-Strategien von zwei Spielern

    :param payoff_matrix_1: Die Auszahlungsmatrix von Spieler 1
    :type payoff_matrix_1: ndarray
    :param payoff_matrix_2: Die Auszahlungsmatrix von Spieler 2 (default: np.array([]))
    :type payoff_matrix_2: ndarray
    :returns: Eine Liste, die eine Liste aller Maximin-Strategien von Spieler 1 und eine Liste aller Maximin-Strategien\
     von Spieler 2 enthält
    :rtype: list
    """
    if not payoff_matrix_2.size:
        payoff_matrix_2 = payoff_matrix_1*-1
    lower_values = get_lower_values(payoff_matrix_1, payoff_matrix_2)
    payoff_matrix_2 = payoff_matrix_2.T
    #minima_player_1 = list()
    #for line in range(payoff_matrix_1.shape[0]):
    #    minima_player_1.append(np.amin(payoff_matrix_1[line][:]))
    minima_player_1 = [np.amin(payoff_matrix_1[line][:]) for line in range(payoff_matrix_1.shape[0])]
    #minima_player_2 = list()
    #for column in range(payoff_matrix_2.shape[1]):
    #    minima_player_2.append(np.amin(payoff_matrix_2[:, column]))
    minima_player_2 = [np.amin(payoff_matrix_2[line][:]) for line in range(payoff_matrix_2.shape[0])]
    #player_1_maximin = list()
    #player_2_maximin = list()

    #for strategy in range(len(minima_player_1)):
    #    if minima_player_1[strategy] == (lower_values[0]):
    #        player_1_maximin.append(strategy)
    #for strategy_2 in range(len(minima_player_2)):
    #    if minima_player_2[strategy_2] == (lower_values[1]):
    #        player_2_maximin.append(strategy_2)
    player_1_maximin = [strategy for strategy in range(len(minima_player_1)) if
                        minima_player_1[strategy] == (lower_values[0])]
    player_2_maximin = [strategy2 for strategy2 in range(len(minima_player_2)) if
                        minima_player_2[strategy2] == (lower_values[1])]
    # print('Minmax-Strategien: ', player_1_maximin, player_2_maximin)

    return [player_1_maximin, player_2_maximin]


# Bayes Strategie von player, wenn der andere Spieler strategy wählt
# Benötigt für Betrachtung Spieler 1 die transponierte Auszahlungsmatrix
# Für Spieler 2 kann die normale Auszahlungsmatrix verwendet werden
def bayes_strategy(payoff_matrix, strategy):
    """Ermittelt bei gegebener gegnerischen Strategie alle Bayes-Strategien

    :param payoff_matrix: Auszahlungsmatrix von Spieler 1 oder transponierte Auszahlungsmatrix von Spieler 2
    :type payoff_matrix: ndarray
    :param strategy: Gespielte Strategie des Gegenspielers
    :type strategy: int
    :returns: Die gefundenen Bayes-Strategien als Liste und die hierfür ausgewerteten Auszahlungen
    :rtype: tuple
    """
    bayes = np.where(payoff_matrix[strategy] == np.max(payoff_matrix[strategy]))
    watched_strategy = payoff_matrix[strategy]
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
# TODO: Simplex-Tableaus graphisch aufbereiten
# TODO: Parametrisierung welche Aufgaben gestellt wurden
def get_calculations_latex(matrix1, matrix2=np.array([]), zerosum=True, bay1=0, bay2=0, mode=0, rand_bays=False):
    """Erzeugt eine verwendbare Ausgabe der berechneten Kennzahlen

    :param matrix1: Die Auszahlungsmatrix von Spieler 1
    :type matrix1: ndarray
    :param matrix2: Die Auszahlungsmatrix von Spieler 2 (default: np.array([]))
    :type matrix2: ndarray
    :param zerosum: Angabe, ob es sich um ein Nullsummenspiel handelt oder nicht (default: False)
    :type zerosum: bool
    :param bay1: Die von Spieler 2 gespielte Strategie zur Ermittlung der Bayes-Strategien (default: 0)
    :type bay1: int
    :param bay2: Die von Spieler 1 gespielte Strategie zur Ermittlung der Bayes-Strategien (default: 0)
    :type bay2: int
    :param mode: Angabe ob Berechnungen nur in reinen Strategien (mode=0) oder auch in gemischten Strategien (mode=1)\
     erfolgen sollen (default: 0)
    :type mode: int
    :param rand_bays: Gibt an, ob zur Ermittlung der Bayes-Strategien zufällige Gegenstrategien gewählt werden sollen
    :type rand_bays: bool
    :returns: Einen String zur Ausgabe, ein für LaTeX formatierter String und ein Context-Dictionary als Liste
    :rtype: list
    """
    if not matrix2.size:
        matrix2 = matrix1*-1
    #sol_tex = []
    sol_texpure = []
    sol_texmixed = []
    matrix1 = np.asarray(matrix1) # Auszahlungsmatrix Spieler 1
    matrix2 = np.asarray(matrix2) # Auszahlungsmatrix Spieler 2
    solution = ''
    context = {} # Rückgabe-Dictionary
    response = {} # Rückgabe-Dictionary neu
    response['matrix1'] = matrix1
    response['matrix2'] = matrix2
    response['zerosum'] = zerosum

    maximins = solve_maximin_strategies(matrix1, matrix2) # Maximin-Strategien ermitteln
    maximins_1 = []
    maximins_2 = []
    for element in maximins[0]:
        maximins_1.append(element+1) # Maxmin-Strategien von Spieler 1
    for element in maximins[1]:
        maximins_2.append(element+1) # Maximin-Strategien von Spieler 2
    response['maximin1'] = maximins_1
    response['maximin2'] = maximins_2
    dets = determination_intervall(matrix1) # Indeterminiertheitsintervall
    response['indet_intervall'] = dets
    guarantee = get_guaranteed_payoff(matrix1, matrix2)
    response['guarantee_point'] = guarantee[0]
    if rand_bays:
        bay2 = np.random.randint(0, matrix1.shape[0]) # Strategie Spieler 1 für Bayes-Strategien Spieler 2
        bay1 = np.random.randint(0, matrix1.shape[1]) # Strategie Spieler 2 für Bayes-Strategien Spieler 1
    response['baystrats1strat2'] = bay2 + 1
    response['baystrats2strat1'] = bay1 + 1
    bs_1 = bayes_strategy(matrix1.T, bay1)
    bayes_1 = bs_1[0] # Bayes-Strategien von Spieler 1
    watch_1 = bs_1[1] # Zu betrachtende Auszahlungen von Spieler 1
    response['baystrats1'] = bayes_1[0]
    response['baystrats1watch1'] = watch_1
    bs_2 = bayes_strategy(matrix2, bay2)
    bayes_2 = bs_2[0] # Bayes-Strategien von Spieler 2
    watch_2 = bs_2[1] # Zu betrachtende Auszahlungen von Spieler 2
    response['baystrats2'] = bayes_2[0]
    response['baystrats2watch2'] = watch_2
    equi = ggw(matrix1, matrix2) # Gleichgewichtspunkte in reinen Strategien
    low_values = get_lower_values(matrix1, matrix2) # Untere Spielwerte
    response['lower_value1'] = low_values[0]
    response['lower_value2'] = low_values[1]
    equi_points = []
    optimals = equi[2] # Auswertungsmatrix zur Bestimmung von Nash-GGW (rein)
    response['nggw_pure_evaluation'] = optimals
    optimals1 = equi[3] # Auswertungsmatrix Spieler 1 zur Bestimmung von Nash-GGW (rein)
    response['nggw_pure_evaluation1'] = optimals1
    optimals2 = equi[4] # Auswertungsmatrix Spieler 2 zur Bestimmung von Nash-GGW (rein)
    response['nggw_pure_evaluation2'] = optimals2

    determined = is_determined(matrix1) # Aussage, ob Spiel determiniert
    response['is_determined'] = determined
    #print(equi)
    #print('equi0')
    #print(equi[0])
    for equis in equi[0]:
        #print(equis)
        if np.asarray(equis).size:
             equi_points.append([equis[0]+1, equis[1]+1]) # Gleichgewichtspunkte als Liste zusammenführen
    response['nggw_pure'] = equi_points
    if len(equi_points) > 0:
        response['pure_nggw'] = True
        response['nggw_pure_payoff1'] = matrix1[equi[0][0][0]][equi[0][0][1]]
        response['nggw_pure_payoff2'] = matrix2[equi[0][0][0]][equi[0][0][1]]
    else:
        response['pure_nggw'] = False
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
    solution += 'ergeben sich folgende Kennzahlen: ' + '\n'
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
    solution += 'Um die Bayes-Strategie zu ermitteln muss die maximale Auszahlung bei gegebener ' + \
    'Gegnerstrategie betrachtet werden.' + '\n'
    solution += 'Für Spieler 1 müssen deshalb bei gegebener Strategie ' + str(bay1+1) + \
                ' von Spieler 2 die Auszahlungen ' + str(watch_1) + ' betrachtet werden.' + '\n'
    #print(bayes_1[0])
    solution += 'Hieraus ergeben sich die Bayes-Strategie(n): ' + str(bayes_1[0]+1) + '\n'
    context['bay1'] = str(bay1+1)
    context['pay1'] = str(watch_1)
    context['baystrat1'] = str(bayes_1[0]+1)
    sol_texpure.append(bay1+1)
    sol_texpure.append(watch_1)
    sol_texpure.append(bayes_1[0]+1)
    solution += 'Für Spieler 2 müssen deshalb bei gegebener Strategie ' + str(
        bay2 + 1) + ' von Spieler 1 die Auszahlungen ' + str(watch_2) + ' betrachtet werden.' + '\n'
    solution += 'Hieraus ergeben sich die Bayes-Strategie(n): ' + str(bayes_2[0]+1) + '\n'
    context['bay2'] = str(bay2 + 1)
    context['pay2'] = str(watch_2)
    context['baystrat2'] = str(bayes_2[0] + 1)
    sol_texpure.append(bay2 + 1)
    sol_texpure.append(watch_2)
    sol_texpure.append(bayes_2[0] + 1)
    solution += 'Das Erfüllen der Optimalitätsbedingung der Strategiekombinationen über beide Spieler ' + \
    'aufsummiert sieht wie folgt aus:' + '\n'
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
        solution += 'Jede Strategiekombination, die sowohl für Spieler 1, als auch für Spieler 2 die ' + \
        'Optimalitätsbedingung erfüllt ist Gleichgewichtspunkt des Spiels in reinen Strategien' + '\n'
        solution += 'Gleichgewichtspunkt(e): ' + str(equi_points) + ' mit zugehöriger Auszahlung für Spieler 1: ' + \
                    str(matrix1[equi[0][0][0]][equi[0][0][1]]) + '\n' + 'und Auszahlung für Spieler 2: ' +\
                    str(matrix2[equi[0][0][0]][equi[0][0][1]])
        sol_texpure.append(equi_points)
        sol_texpure.append(matrix1[equi[0][0][0]][equi[0][0][1]])
        sol_texpure.append(matrix2[equi[0][0][0]][equi[0][0][1]])
        context['puresolve'] = 'Jede Strategiekombination, die sowohl für Spieler 1, als auch für Spieler 2 die \
        Optimalitätsbedingung erfüllt ist Gleichgewichtspunkt des Spiels in reinen Strategien\\Gleichgewichtspunkt(e): '\
                               + str(equi_points) + ' mit zugehöriger Auszahlung für Spieler 1: ' + \
                               str(matrix1[equi[0][0][0]][equi[0][0][1]]) + '\n' + 'und Auszahlung für Spieler 2: ' + \
                               str(matrix2[equi[0][0][0]][equi[0][0][1]]) + r'\\'
    else:
        solution += 'Da keine Strategiekombination sowohl für Spieler 1, als auch für Spieler 2 die ' \
                    'Optimalitätsbedingung erfüllt existiert kein Gleichgewichtspunkt in reinen Strategien'
        sol_texpure.append([])
        sol_texpure.append([])
        sol_texpure.append([])
        context['puresolve'] = 'Da keine Strategiekombination sowohl für Spieler 1, als auch für Spieler 2 die ' \
                               'Optimalitätsbedingung erfüllt existiert kein Gleichgewichtspunkt in reinen Strategien\\'
    context['solvemixed'] = ""

    if mode > 0:
        response['evaluate_mixed'] = True
    else:
        response['evaluate_mixed'] = False

    if mode > 0:
        context['solvemixed'] = ''
        if zerosum:
            simplex = use_simplex(matrix1, matrix2)
            if not np.array_equal(simplex[2][0], matrix1):
                response['c_added'] = True
                response['transformed_matrix'] = simplex[2][0]
                solution += 'Aufgrund der Beschränkungen des Simplex-Algorithmus muss zunächst die ' \
                            'Auszahlungsmatrix des betrachteten Zwei-Personen-Nullsummenspiels absolut ' \
                            'positiv werden.' + '\n'
                solution += 'Die zu lösende Matrix sieht nun folgendermaßen aus: ' + '\n'
                solution += str(simplex[2][0]) + '\n'
                sol_texmixed.append(simplex[2][0])
                context['solvemixed'] += r'Aufgrund der Beschränkungen des Simplex-Algorithmus muss ' \
                                         r'zunächst die Auszahlungsmatrix des betrachteten ' \
                                         r'Zwei-Personen-Nullsummenspiels absolut positiv werden.\\Die zu ' \
                                         r'lösende Matrix sieht nun folgendermaßen aus:\\'
                context['solvemixed'] += r'\begin{gather*}\begin{pmatrix*}'
                temp_str = ''
                for lines in range(simplex[2][0].shape[0]):
                    for cols in range(simplex[2][0].shape[1]):
                        temp_str += str(simplex[2][0][lines][cols]) + '&'
                    temp_str += r'\\'
                context['solvemixed'] += temp_str + '\end{pmatrix*}\end{gather*}'
            else:
                response['c_added'] = False
                solution += 'Das folgende Spiel soll nun mithilfe des Simplex-Algorithmus gelöst werden:' + '\n'
                solution += str(matrix1)
                sol_texmixed.append(matrix1)
                context['solvemixed'] += r'Das folgende Spiel soll nun mithilfe des Simplex-Algorithmus ' \
                                         r'gelöst werden:\\'
                temp_str = ''
                for lines in range(matrix1.shape[0]):
                    for cols in range(matrix1.shape[1]):
                        temp_str += str(matrix1[lines][cols]) + '&'
                    temp_str += r'\\'
                context['solvemixed'] += temp_str + r'\\'
            solution += 'Die Lösungsschritte des Simplexalgorithmus für Spieler 2 sehen nun wie folgt aus: ' + '\n'
            context['solvemixed'] += r'Die Lösungsschritte des Simplexalgorithmus für Spieler 2 sehen nun wie ' \
                                     r'folgt aus:\\ ' + r' \begin{gather*}'
            #context['solvemixed'] += r'\begin{gather*}'
            #response['simplex_steps'] = simplex[1][1:][0]
            tempo_full = []
            for step in simplex[1][1:][0]:
                tempo = []
                temp_arr = np.asarray(format_solution(step['tableau']))
                tempo.append(temp_arr)
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
                    tempo.append([step['pivot'][1], step['pivot'][0]])
                    context['solvemixed'] += 'Pivot: ' + str(step['pivot']) + r'\\'
                else:
                    sol_texmixed.append([])
                    tempo.append([])
                tempo_full.append(deepcopy(tempo))
            response['simplex_steps'] = deepcopy(tempo_full[1:])
            response['first_step'] = deepcopy(tempo_full[0])
            context['solvemixed'] += r'\end{gather*}\\'
            added_value = np.amax(simplex[2][0]-matrix1)
            response['added_value'] = added_value
            solution += 'Da nicht der Spielwert maximiert wurde, sondern 1/G minimiert wurde und eine ' \
                        'Konstante ' + str(added_value) + ' zur Matrix addiert wurde, muss man die ' \
                                                          'Konstante wieder vom Ergebnis subtrahieren und den ' \
                                                          'Kehrbruch verwenden.' + '\n'
            sol_texmixed.append(added_value)
            context['solvemixed'] += r'Da nicht der Spielwert maximiert wurde, sondern 1/G minimiert wurde und ' \
                                     r'eine Konstante ' + str(added_value) + r' zur Matrix addiert wurde, muss ' \
                                                                             r'man die Konstante wieder vom ' \
                                                                             r'Ergebnis subtrahieren und den ' \
                                                                             r'Kehrbruch verwenden.\\'
            game_value_1 = nsimplify((1/simplex[1][0]['fun']) + added_value, tolerance=0.0001, rational=True)
            strategies = []
            for strategy in simplex[1][0]['x']:
                strategies.append(nsimplify(abs(((1/simplex[1][0]['fun'])*strategy)),
                                            tolerance=0.0001, rational=True))
            solution += 'Hieraus ergibt sich der tatsächliche Spielwert für Spieler 2: ' + str(game_value_1) + '\n'
            response['game_value_simplex_player2'] = game_value_1
            context['solvemixed'] += 'Hieraus ergibt sich der tatsächliche Spielwert für Spieler 2: ' + \
                                     str(game_value_1) + r'\\'
            sol_texmixed.append(game_value_1)
            solution += 'Und die optimale Strategienkombination für Spieler 2: ' + str(strategies) + '\n'
            context['solvemixed'] += 'Und die optimale Strategienkombination für Spieler 2: ' + \
                                     str(strategies) + r'\\'
            sol_texmixed.append(strategies)
            response['optimal_strategies_simplex_player2'] = strategies
            solution += 'Da ein Zwei-Personen-Nullsummenspiel betrachtet wurde ergibt sich der Spielwert ' \
                        'für Spieler 1: ' + str(game_value_1*-1) + '\n'
            context['solvemixed'] += 'Da ein Zwei-Personen-Nullsummenspiel betrachtet wurde ergibt sich der ' \
                                     'Spielwert für Spieler 1: ' + str(game_value_1*-1) + r'\\'
            response['game_value_simplex_player1'] = game_value_1*-1
            strategies_1 = []
            sol_texmixed.append(game_value_1*-1)
            possible_choices = -1 - matrix1.shape[0]
            for elements in simplex[1][1][-1]['tableau'][-1][possible_choices:-1]:
                strategies_1.append(nsimplify(elements*(1/simplex[1][1][-1]['tableau'][-1][-1]),
                                              tolerance=0.0001, rational=True))
            solution += 'Die Dualität des Problems erlaubt es, die optimale Strategienkombination für Spieler 1' \
                        ' direkt aus der Zielfunktionszeile abzulesen: ' + str(strategies_1) + '\n'
            response['optimal_strategies_simplex_player1'] = strategies_1
            sol_texmixed.append(strategies_1)
            context['solvemixed'] += 'Die Dualität des Problems erlaubt es, die optimale Strategienkombination' \
                                     ' für Spieler 1 direkt aus der Zielfunktionszeile abzulesen: ' + \
                                     str(strategies_1) + r'\\'
            solution += '\n\n\n'
            reduced = reduce_matrix(matrix1, matrix2)
            #boole = reduced[2]
            #if boole:
            #    solution += 'Zunächst müssen die überflüssigen Zeilen und Spalten der Ausgangsmatrix
            # entfernt werden.' + '\n'
            #    solution += 'Hierdurch ergibt sich folgende Matrix für Spieler 1:' + '\n'
            #    solution += str(reduced[0]) + '\n'

            # try:
            #     nggw = solve_using_nggw(matrix1, matrix2)
            #     nggw2 = get_optimal_solution(matrix1, matrix2)
            #     solution_1 = nggw[0]
            #     lgs1 = solution_1[2]
            #     solution_2 = nggw[1]
            #     lgs2 = solution_2[2]
            #     correct_solution = True
            #     for key in solution_1[1]:
            #         if solution_1[0][key] < 0 and key != solution_1[1][-1]:
            #             correct_solution = False
            #             solution += 'Key-Fehler: ' + str(solution_1[0][key]) + ' ' + str(key) + '\n'
            #     for key in solution_2[1]:
            #         if solution_2[0][key] < 0 and key != solution_2[1][-1]:
            #             correct_solution = False
            #             solution += 'Key-Fehler: ' + str(solution_2[0][key]) + ' ' + str(key) + '\n'
            #     if correct_solution:
            #         solution += 'Selbiges Problem lässt sich auch durch die Aufstellung eines LGS nach den ' \
            #                     'Bedingungen für ein Nash-Gleichgewicht lösen: ' + '\n'
            #         context['solvemixed'] += r'Selbiges Problem lässt sich auch durch die Aufstellung eines ' \
            #                                  r'LGS nach den Bedingungen für ein Nash-Gleichgewicht lösen:\\'
            #         if len(solution_1[0]) > 0:
            #             solution += 'Das lineare Gleichungssystem für Spieler 1 lautet: ' + '\n'
            #             context['solvemixed'] += r'Das lineare Gleichungssystem für Spieler 1 lautet:\\'
            #             temp = []
            #             for equation in lgs1:
            #                 solution += str(equation)
            #                 solution += '\n'
            #                 temp.append(equation)
            #                 context['solvemixed'] += str(equation) + r'\\'
            #             sol_texmixed.append(temp)
            #             solution += 'Nach Auflösen des LGS ergeben sich folgene Werte für die optimale ' \
            #                         'Strategienkombination: ' + '\n'
            #             context['solvemixed'] += r'Nach Auflösen des LGS ergeben sich folgene Werte für ' \
            #                                      r'die optimale Strategienkombination:\\'
            #             temp = []
            #             for key in solution_1[1]:
            #                 solution += str(key) + ':' + str(nsimplify(solution_1[0][key], tolerance=0.0001,
            #                                                            rational=True)) + '\n'
            #                 val = nsimplify(solution_1[0][key], tolerance=0.0001, rational=True)
            #                 temp.append([key, val])
            #                 context['solvemixed'] += str(key) + ':' + str(val) + r'\\'
            #             sol_texmixed.append(temp)
            #         if len(solution_2[0]) > 0:
            #             solution += 'Das lineare Gleichungssystem für Spieler 2 lautet: ' + '\n'
            #             context['solvemixed'] += r'Das lineare Gleichungssystem für Spieler 2 lautet:\\'
            #             temp = []
            #             for equation in lgs2:
            #                 solution += str(equation)
            #                 temp.append(equation)
            #                 solution += '\n'
            #                 context['solvemixed'] += str(equation) + r'\\'
            #             sol_texmixed.append(temp)
            #             solution += 'Nach Auflösen des LGS ergeben sich folgene Werte für die optimale ' \
            #                         'Strategienkombination: ' + '\n'
            #             context[
            #                 'solvemixed'] += r'Nach Auflösen des LGS ergeben sich folgene Werte für die ' \
            #                                  r'optimale Strategienkombination:\\'
            #             temp = []
            #             for key in solution_2[1]:
            #                 solution += str(key) + ':' + str(nsimplify(solution_2[0][key], tolerance=0.0001,
            #                                                            rational=True)) + '\n'
            #                 val = nsimplify(solution_2[0][key], tolerance=0.0001, rational=True)
            #                 temp.append([key, val])
            #                 context['solvemixed'] += str(key) + ':' + str(val) + r'\\'
            #             sol_texmixed.append(temp)
            #     solution += nggw2
            # except:
            #     pass
            nggw2 = get_optimal_solution(matrix1, matrix2)
            solution += 'Selbiges Problem lässt sich auch durch die Aufstellung eines LGS nach den ' \
                        'Bedingungen für ein Nash-Gleichgewicht lösen: ' + '\n'
            context['solvemixed'] += r'Selbiges Problem lässt sich auch durch die Aufstellung eines ' \
                                     r'LGS nach den Bedingungen für ein Nash-Gleichgewicht lösen:\\'
            symbs = generate_symbols(matrix1)
            functs = generate_functions(matrix1, matrix2, symbs=symbs)
            solution += 'Das lineare Gleichungssystem für die Wahrscheinlichkeiten der Strategien von ' \
                        'Spieler 1 und der Auszahlung von Spieler 2 lautet: ' + '\n'
            context['solvemixed'] += r'Das lineare Gleichungssystem für die Wahrscheinlichkeiten der ' \
                                     r'Strategeien von Spieler 1 und der Auszahlung von Spieler 2 lautet:\\'
            temp = []
            temp2 = []
            for equation in functs[0]:
                solution += str(equation) + ' = 0'
                solution += '\n'
                temp.append(equation)
                context['solvemixed'] += str(equation) + ' = 0' + r'\\'
                temp2.append(str(equation) + ' = 0')
            sol_texmixed.append(temp)
            response['lgs_player1'] = temp2 # Auszahlung Spieler 2 und Wahrscheinlichkeiten Spieler 1

            solution += 'Das lineare Gleichungssystem für die Wahrscheinlichkeiten der Strategien von ' \
                        'Spieler 2 und der Auszahlung von Spieler 1 lautet: ' + '\n'
            context['solvemixed'] += r'Das lineare Gleichungssystem für die Wahrscheinlichkeiten der ' \
                                     r'Strategeien von Spieler 2 und der Auszahlung von Spieler 1 lautet:\\'
            temp = []
            temp2 = []
            for equation in functs[1]:
                solution += str(equation) + ' = 0'
                temp2.append(str(equation) + ' = 0')
                solution += '\n'
                temp.append(equation)
                context['solvemixed'] += str(equation) + ' = 0' + r'\\'
            sol_texmixed.append(temp)
            response['lgs_player2'] = temp2  # Auszahlung Spieler 1 und Wahrscheinlichkeiten Spieler 2
            solution += 'Es müssen nun alle möglichen Support-Kombinationen der beiden Spieler betrachtet werden,' \
                        ' wobei folgende Support-Kombinationen zu einem Nash-Gleichgewicht führen:' + '\n'
            context['solvemixed'] += r'Es müssen nun alle möglichen Support-Kombinationen der beiden Spieler ' \
                                     r'betrachtet werden, wobei folgende Support-Kombinationen zu einem ' \
                                     r'Nash-Gleichgewicht führen:\\'
            supps = []
            response['supports'] = []
            # response['support_player2'] = []
            response['lgs_support_player1'] = []
            response['lgs_support_player2'] = []
            response['lgs_support_results_player1'] = []
            response['lgs_support_results_player2'] = []
            response['lgs_support_game_value1'] = []
            response['lgs_support_game_value2'] = []
            response['results_out_of_support'] = []
            response['lgs_nonsupport_lgs_player1'] = []
            response['lgs_nonsupport_lgs_player2'] = []
            response['lgs_nonsupport_result_player1'] = []
            response['lgs_nonsupport_result_player2'] = []
            response['mixed_ggw'] = []
            for mixed_ggw in nggw2:
                supps.append([mixed_ggw[0][0][2], mixed_ggw[0][1][2]])
                temp = []
                temp.append([mixed_ggw[0][0][2], mixed_ggw[0][1][2]])
                response['supports'].append([mixed_ggw[0][0][2], mixed_ggw[0][1][2]])
                # supps.add(mixed_ggw[0][1][2])
                solution += str(mixed_ggw[0][0][2]) + ', ' + str(mixed_ggw[0][1][2]) + '\n'
                solution += 'Und somit die zugehörigen Gleichungssystemen betrachtet werden:\n'
                context['solvemixed'] += str(mixed_ggw[0][0][2]) + ', ' + str(mixed_ggw[0][1][2]) + r'\\'
                context['solvemixed'] += r'Und somit die zugehörigen Gleichungssystemen betrachtet werden:\\'
                temp2 = []
                for eq in mixed_ggw[0][2][0][0]:
                    solution += str(eq) + ' = 0\n'
                    temp2.append(str(eq) + ' = 0')
                    context['solvemixed'] += str(eq) + ' = 0' +r'\\'
                solution += '\n'
                response['lgs_support_player1'].append(mixed_ggw[0][2][0][0])
                temp.append(temp2)
                context['solvemixed'] += r'\\'
                temp2 = []
                for eq in mixed_ggw[0][2][1][0]:
                    solution += str(eq) + ' = 0\n'
                    temp2.append(str(eq) + ' = 0')
                    context['solvemixed'] += str(eq) + r' = 0\\'
                response['lgs_support_player2'].append(mixed_ggw[0][2][1][0])
                temp.append(temp2)

                solution += 'Nach Auflösen des LGS ergeben sich folgene Werte für die optimale ' \
                        'Strategienkombination: ' + '\n'
                context['solvemixed'] += r'Nach Auflösen des LGS ergeben sich folgene Werte für die ' \
                                     r'optimale Strategienkombination:\\'
                solution += 'Für Spieler 1:\n'
                context['solvemixed'] += r'Für Spieler 1:\\'
                temp_sol = {}
                temp2 = []
                for key in symbs[0][0]:
                    solution += str(key) + ' = ' + str(mixed_ggw[0][0][1][0][key]) + '\n'
                    temp2.append(str(key) + ' = ' + str(mixed_ggw[0][0][1][0][key]))
                    context['solvemixed'] += str(key) + ' = ' + str(mixed_ggw[0][0][1][0][key]) + r'\\'
                    temp_sol[key] = mixed_ggw[0][0][1][0][key]
                response['lgs_support_results_player1'].append(temp_sol)
                temp.append(temp2)
                solution += 'Für Spieler 2:\n'
                context['solvemixed'] += r'Für Spieler 2:\\'
                temp_sol = {}
                temp2 = []
                for key in symbs[1][0]:
                    solution += str(key) + ' = ' + str(mixed_ggw[0][1][1][0][key]) + '\n'
                    context['solvemixed'] += str(key) + ' = ' + str(mixed_ggw[0][1][1][0][key]) + r'\\'
                    temp_sol[key] = mixed_ggw[0][1][1][0][key]
                    temp2.append(str(key)+ ' = ' + str(mixed_ggw[0][1][1][0][key]))
                response['lgs_support_results_player2'].append(temp_sol)
                temp.append(temp2)

                solution += 'Der Spielwert von Spieler 1 entspricht: ' + '\n'
                context['solvemixed'] += r'Der Spielwert von Spieler 1 entspricht: \\'
                solution += 'w = ' + str(mixed_ggw[0][0][0]) + '\n'
                response['lgs_support_game_value1'] = mixed_ggw[0][0][0]
                temp.append(mixed_ggw[0][0][0])
                context['solvemixed'] += 'w = ' + str(mixed_ggw[0][0][0]) + r'\\'
                solution += 'Der Spielwert von Spieler 2 entspricht: ' + '\n'
                context['solvemixed'] += r'Der Spielwert von Spieler 2 entspricht: \\'
                solution += 'w = ' + str(mixed_ggw[0][1][0]) + '\n'
                response['lgs_support_game_value2'] = mixed_ggw[0][1][0]
                temp.append(mixed_ggw[0][1][0])
                context['solvemixed'] += 'w = ' + str(mixed_ggw[0][1][0]) + r'\\'
                responses = []
                if len(mixed_ggw[0][3][0]) or len(mixed_ggw[0][3][1]):
                    #response['results_out_of_support'].append(True)
                    solution += 'Nun müssen die erzielbaren Auszahlungen eines Spielers, durch die Wahl einer ' \
                            'Strategie, die nicht im betrachteten Support liegt ermittelt werden.\n'
                    context['solvemixed'] += r'Nun müssen die erzielbaren Auszahlungen eines Spielers, durch die ' \
                                         r'Wahl einer Strategie, die nicht im betrachteten Support liegt ' \
                                         r'ermittelt werden.\\'
                    if len(mixed_ggw[0][3][1]):
                        responses.append(True)
                        temp.append(True)
                        solution += 'Für die Auszahlung von Spieler 1 werden somit folgende Gleichungssysteme ' \
                                    'und Lösungen betrachtet: \n'
                        context['solvemixed'] += r'Für die Auszahlung von Spieler 1 werden somit folgende ' \
                                                 r'Gleichungssysteme und Lösungen betrachtet: \\'
                        del tempo[:]
                        game_val_mixed_1 = []
                        strats = []
                        tempo = []
                        for notsupportstrat in mixed_ggw[0][3][1]:
                            temp2 = []
                            for eq in notsupportstrat[0]:
                                solution += str(eq) + ' = 0\n'
                                temp2.append(str(eq) + ' = 0')
                                context['solvemixed'] += str(eq) + r' = 0\\'
                            solution += 'Dieses Gleichungssystem führt zu einer Auszahlung für Spieler 1 von:\n'
                            context['solvemixed'] += r'Dieses Gleichungssystem führt zu einer Auszahlung für ' \
                                                     r'Spieler 1 von:\\'
                            solution += 'w = ' + str(notsupportstrat[1]) + '\n'
                            context['solvemixed'] += 'w = ' + str(notsupportstrat[1]) + r'\\'
                            solution += 'Was dazu führt, dass eine Abweichung von den Support-Strategien für ' \
                                        'Spieler 1 keine Auszahlungsverbesserung hervorruft.\n'
                            context['solvemixed'] += r'Was dazu führt, dass eine Abweichung von den ' \
                                                     r'Support-Strategien für Spieler 1 keine ' \
                                                     r'Auszahlungsverbesserung hervorruft.\\'
                            game_val_mixed_1.append(notsupportstrat[1])
                            strats.append(temp2)
                            tempo.append([temp2, notsupportstrat[1]])
                        response['lgs_nonsupport_lgs_player1'].append(strats)
                        response['lgs_nonsupport_result_player1'].append(game_val_mixed_1)
                        temp.append(deepcopy(tempo))
                    else:
                        responses.append(False)
                        temp.append(False)
                        solution += 'Für Spieler 1 liegen alle Strategien in der betrachteten Support-Menge, ' \
                                    'wodurch keine weitere Betrachtung notwendig ist.\n'
                        context['solvemixed'] += r'Für Spieler 1 liegen alle Strategien in der betrachteten ' \
                                                 r'Support-Menge, wodurch keine weitere Betrachtung notwendig ist.\\'
                        response['lgs_nonsupport_lgs_player1'].append([])
                        response['lgs_nonsupport_result_player1'].append([])
                        temp.append([[],[]])
                    if len(mixed_ggw[0][3][0]):
                        responses.append(True)
                        temp.append(True)
                        solution += 'Für die Auszahlung von Spieler 2 werden somit folgende Gleichungssysteme ' \
                                    'und Lösungen betrachtet: \n'
                        context['solvemixed'] += r'Für die Auszahlung von Spieler 2 werden somit folgende ' \
                                                 r'Gleichungssysteme und Lösungen betrachtet: \\'

                        game_val_mixed_2 = []
                        strats = []
                        del tempo[:]
                        tempo = []
                        for notsupportstrat in mixed_ggw[0][3][0]:
                            temp2 = []
                            for eq in notsupportstrat[0]:
                                solution += str(eq) + ' = 0\n'
                                temp2.append(str(eq) + ' = 0')
                                context['solvemixed'] += str(eq) + r' = 0\\'
                            solution += 'Dieses Gleichungssystem führt zu einer Auszahlung für Spieler 2 von:\n'
                            context['solvemixed'] += r'Dieses Gleichungssystem führt zu einer Auszahlung für ' \
                                                     r'Spieler 2 von:\\'
                            solution += 'w = ' + str(notsupportstrat[1]) + '\n'
                            context['solvemixed'] += 'w = ' + str(notsupportstrat[1]) + r'\\'
                            solution += 'Was dazu führt, dass eine Abweichung von den Support-Strategien für ' \
                                        'Spieler 2 keine Auszahlungsverbesserung hervorruft.\n'
                            context['solvemixed'] += r'Was dazu führt, dass eine Abweichung von den ' \
                                                     r'Support-Strategien für Spieler 2 keine ' \
                                                     r'Auszahlungsverbesserung hervorruft.\\'
                            game_val_mixed_2.append(notsupportstrat[1])
                            strats.append(temp2)
                            tempo.append([temp2, notsupportstrat[1]])
                        response['lgs_nonsupport_lgs_player2'].append(strats)
                        response['lgs_nonsupport_result_player2'].append(game_val_mixed_2)
                        temp.append(deepcopy(tempo))
                    else:
                        responses.append(False)
                        temp.append(False)
                        solution += 'Für Spieler 2 liegen alle Strategien in der betrachteten Support-Menge, ' \
                                    'wodurch keine weitere Betrachtung notwendig ist.\n'
                        context['solvemixed'] += r'Für Spieler 2 liegen alle Strategien in der betrachteten ' \
                                                 r'Support-Menge, wodurch keine weitere Betrachtung notwendig ist.\\'
                        response['lgs_nonsupport_lgs_player2'].append([])
                        temp.append([[],[]])
                        response['lgs_nonsupport_result_player2'].append([])
                else:
                    solution += 'Da für beide Spieler keine Strategie nicht in der Support-Menge liegt, ' \
                                'müssen keine weiteren Berechnungen durchgeführt werden.\n'
                    context['solvemixed'] += r'Da für beide Spieler keine Strategie nicht in der Support-Menge ' \
                                             r'liegt, müssen keine weiteren Berechnungen durchgeführt werden.\\'
                    responses = [False, False]
                    temp.append(False)
                    temp.append([[],[]])
                    temp.append(False)
                    temp.append([[],[]])
                response['results_out_of_support'].append(responses)
                response['mixed_ggw'].append(temp)
            # for mixed_ggw in nggw2:
            #     solution += '\n'
            #     solution += str(mixed_ggw)
            #     solution += '\n'
            #     solution += str(mixed_ggw[0])
            #     solution += '\n'
            #     # Spielwert Spieler 1
            #     solution += str(mixed_ggw[0][0][0])
            #     solution += '\n'
            #     # Wahrscheinlichkeiten Spieler 1
            #     solution += str(mixed_ggw[0][0][1][0])
            #     solution += '\n'
            #     # Support Spieler 1
            #     solution += str(mixed_ggw[0][0][2])
            #     solution += '\n'
            #     # Spielwert Spieler 2
            #     solution += str(mixed_ggw[0][1][0])
            #     solution += '\n'
            #     # Wahrscheinlichkeiten Spieler 2
            #     solution += str(mixed_ggw[0][1][1][0])
            #     solution += '\n'
            #     # Support Spieler 2
            #     solution += str(mixed_ggw[0][1][2])
            #     solution += '\n'
            #     # Gleichtungssystem Spieler 1
            #     solution += str(mixed_ggw[0][2][0])
            #     solution += '\n'
            #     # Gleichtungssystem Spieler 2
            #     solution += str(mixed_ggw[0][2][1])
            #     solution += '\n'
            #     solution += str(mixed_ggw[0][3][0])
            #     solution += '\n'
            #     solution += str(mixed_ggw[0][3][1])
        else:
            # try:
            #     nggw = solve_using_nggw(matrix1, matrix2)
            #     nggw2 = get_optimal_solution(matrix1, matrix2)
            #     solution_1 = nggw[0]
            #     lgs1 = solution_1[2]
            #     solution_2 = nggw[1]
            #     lgs2 = solution_2[2]
            #     correct_solution = True
            #     for key in solution_1[1]:
            #         if solution_1[0][key] < 0 and key != solution_1[1][-1]:
            #             correct_solution = False
            #             solution += 'Key-Fehler: ' + str(solution_1[0][key]) + ' ' + str(key)+ '\n'
            #     for key in solution_2[1]:
            #         if solution_2[0][key] < 0 and key != solution_2[1][-1]:
            #             correct_solution = False
            #             solution += 'Key-Fehler: ' + str(solution_2[0][key]) + ' ' + str(key) + '\n'
            #     if correct_solution:
            #         solution += 'Selbiges Problem lässt sich auch durch die Aufstellung eines LGS nach den ' \
            #                     'Bedingungen für ein Nash-Gleichgewicht lösen: ' + '\n'
            #         context['solvemixed'] += r'Selbiges Problem lässt sich auch durch die Aufstellung eines ' \
            #                                  r'LGS nach den Bedingungen für ein Nash-Gleichgewicht lösen:\\'
            #         if len(solution_1[0]) > 0:
            #             solution += 'Das lineare Gleichungssystem für Spieler 1 lautet: ' + '\n'
            #             context['solvemixed'] += r'Das lineare Gleichungssystem für Spieler 1 lautet:\\'
            #             temp = []
            #             for equation in lgs1:
            #                 solution += str((equation))
            #                 solution += '\n'
            #                 temp.append(equation)
            #                 context['solvemixed'] += str(equation) + r'\\'
            #             sol_texmixed.append(temp)
            #             solution += 'Nach Auflösen des LGS ergeben sich folgene Werte für die optimale ' \
            #                         'Strategienkombination: ' + '\n'
            #             context['solvemixed'] += r'Nach Auflösen des LGS ergeben sich folgene Werte für die ' \
            #                                      r'optimale Strategienkombination:\\'
            #             temp = []
            #             for key in solution_1[1]:
            #                 solution += str(key) + ':' + str(nsimplify(solution_1[0][key], tolerance=0.0001,
            #                                                            rational=True)) + '\n'
            #                 val = nsimplify(solution_1[0][key], tolerance=0.0001, rational=True)
            #                 temp.append([key, val])
            #                 context['solvemixed'] += str(key) + ':' + str(val) + r'\\'
            #             sol_texmixed.append(temp)
            #         if len(solution_2[0]) > 0:
            #             solution += 'Das lineare Gleichungssystem für Spieler 2 lautet: ' + '\n'
            #             context['solvemixed'] += r'Das lineare Gleichungssystem für Spieler 2 lautet:\\'
            #             temp = []
            #             for equation in lgs2:
            #                 solution += str(equation)
            #                 temp.append(equation)
            #                 solution += '\n'
            #                 context['solvemixed'] += str(equation) + r'\\'
            #             sol_texmixed.append(temp)
            #             solution += 'Nach Auflösen des LGS ergeben sich folgene Werte für die optimale ' \
            #                         'Strategienkombination: ' + '\n'
            #             context[
            #                 'solvemixed'] += r'Nach Auflösen des LGS ergeben sich folgene Werte für die ' \
            #                                  r'optimale Strategienkombination:\\'
            #             temp = []
            #             for key in solution_2[1]:
            #                 solution += str(key) + ':' + str(nsimplify(solution_2[0][key], tolerance=0.0001,
            #                                                            rational=True)) + '\n'
            #                 val = nsimplify(solution_2[0][key], tolerance=0.0001, rational=True)
            #                 temp.append([key, val])
            #                 context['solvemixed'] += str(key) + ':' + str(val) + r'\\'
            #             sol_texmixed.append(temp)
            #     solution += nggw2
            # except:
            #     pass
            nggw2 = get_optimal_solution(matrix1, matrix2)
            solution += 'Selbiges Problem lässt sich auch durch die Aufstellung eines LGS nach den ' \
                        'Bedingungen für ein Nash-Gleichgewicht lösen: ' + '\n'
            context['solvemixed'] += r'Selbiges Problem lässt sich auch durch die Aufstellung eines ' \
                                     r'LGS nach den Bedingungen für ein Nash-Gleichgewicht lösen:\\'
            symbs = generate_symbols(matrix1)
            functs = generate_functions(matrix1, matrix2, symbs=symbs)
            solution += 'Das lineare Gleichungssystem für die Wahrscheinlichkeiten der Strategien von ' \
                        'Spieler 1 und der Auszahlung von Spieler 2 lautet: ' + '\n'
            context['solvemixed'] += r'Das lineare Gleichungssystem für die Wahrscheinlichkeiten der ' \
                                     r'Strategeien von Spieler 1 und der Auszahlung von Spieler 2 lautet:\\'
            temp = []
            for equation in functs[0]:
                solution += str(equation) + ' = 0'
                solution += '\n'
                temp.append(equation)
                context['solvemixed'] += str(equation) + ' = 0' + r'\\'
            sol_texmixed.append(temp)
            response['lgs_player1'] = functs[0]  # Auszahlung Spieler 2 und Wahrscheinlichkeiten Spieler 1

            solution += 'Das lineare Gleichungssystem für die Wahrscheinlichkeiten der Strategien von ' \
                        'Spieler 2 und der Auszahlung von Spieler 1 lautet: ' + '\n'
            context['solvemixed'] += r'Das lineare Gleichungssystem für die Wahrscheinlichkeiten der ' \
                                     r'Strategeien von Spieler 2 und der Auszahlung von Spieler 1 lautet:\\'
            temp = []
            for equation in functs[1]:
                solution += str(equation) + ' = 0'
                solution += '\n'
                temp.append(equation)
                context['solvemixed'] += str(equation) + ' = 0' + r'\\'
            sol_texmixed.append(temp)
            response['lgs_player2'] = functs[1]  # Auszahlung Spieler 1 und Wahrscheinlichkeiten Spieler 2
            solution += 'Es müssen nun alle möglichen Support-Kombinationen der beiden Spieler betrachtet werden,' \
                        ' wobei folgende Support-Kombinationen zu einem Nash-Gleichgewicht führen:' + '\n'
            context['solvemixed'] += r'Es müssen nun alle möglichen Support-Kombinationen der beiden Spieler ' \
                                     r'betrachtet werden, wobei folgende Support-Kombinationen zu einem ' \
                                     r'Nash-Gleichgewicht führen:\\'
            supps = []
            response['supports'] = []
            # response['support_player2'] = []
            response['lgs_support_player1'] = []
            response['lgs_support_player2'] = []
            response['lgs_support_results_player1'] = []
            response['lgs_support_results_player2'] = []
            response['lgs_support_game_value1'] = []
            response['lgs_support_game_value2'] = []
            response['results_out_of_support'] = []
            response['lgs_nonsupport_lgs_player1'] = []
            response['lgs_nonsupport_lgs_player2'] = []
            response['lgs_nonsupport_result_player1'] = []
            response['lgs_nonsupport_result_player2'] = []
            response['mixed_ggw'] = []
            for mixed_ggw in nggw2:
                supps.append([mixed_ggw[0][0][2], mixed_ggw[0][1][2]])
                temp = []
                temp.append([mixed_ggw[0][0][2], mixed_ggw[0][1][2]])
                response['supports'].append([mixed_ggw[0][0][2], mixed_ggw[0][1][2]])
                # supps.add(mixed_ggw[0][1][2])
                solution += str(mixed_ggw[0][0][2]) + ', ' + str(mixed_ggw[0][1][2]) + '\n'
                solution += 'Und somit die zugehörigen Gleichungssystemen betrachtet werden:\n'
                context['solvemixed'] += str(mixed_ggw[0][0][2]) + ', ' + str(mixed_ggw[0][1][2]) + r'\\'
                context['solvemixed'] += r'Und somit die zugehörigen Gleichungssystemen betrachtet werden:\\'
                for eq in mixed_ggw[0][2][0][0]:
                    solution += str(eq) + ' = 0\n'
                    context['solvemixed'] += str(eq) + ' = 0' + r'\\'
                solution += '\n'
                response['lgs_support_player1'].append(mixed_ggw[0][2][0][0])
                temp.append(mixed_ggw[0][2][0][0])
                context['solvemixed'] += r'\\'
                for eq in mixed_ggw[0][2][1][0]:
                    solution += str(eq) + ' = 0\n'
                    context['solvemixed'] += str(eq) + r' = 0\\'
                response['lgs_support_player2'].append(mixed_ggw[0][2][1][0])
                temp.append(mixed_ggw[0][2][1][0])

                solution += 'Nach Auflösen des LGS ergeben sich folgene Werte für die optimale ' \
                            'Strategienkombination: ' + '\n'
                context['solvemixed'] += r'Nach Auflösen des LGS ergeben sich folgene Werte für die ' \
                                         r'optimale Strategienkombination:\\'
                solution += 'Für Spieler 1:\n'
                context['solvemixed'] += r'Für Spieler 1:\\'
                temp_sol = {}
                for key in symbs[0][0]:
                    solution += str(key) + ' = ' + str(mixed_ggw[0][0][1][0][key]) + '\n'
                    context['solvemixed'] += str(key) + ' = ' + str(mixed_ggw[0][0][1][0][key]) + r'\\'
                    temp_sol[key] = mixed_ggw[0][0][1][0][key]
                response['lgs_support_results_player1'].append(temp_sol)
                temp.append(temp_sol)
                solution += 'Für Spieler 2:\n'
                context['solvemixed'] += r'Für Spieler 2:\\'
                temp_sol = {}
                for key in symbs[1][0]:
                    solution += str(key) + ' = ' + str(mixed_ggw[0][1][1][0][key]) + '\n'
                    context['solvemixed'] += str(key) + ' = ' + str(mixed_ggw[0][1][1][0][key]) + r'\\'
                    temp_sol[key] = mixed_ggw[0][1][1][0][key]
                response['lgs_support_results_player2'].append(temp_sol)
                temp.append(temp_sol)

                solution += 'Der Spielwert von Spieler 1 entspricht: ' + '\n'
                context['solvemixed'] += r'Der Spielwert von Spieler 1 entspricht: \\'
                solution += 'w = ' + str(mixed_ggw[0][0][0]) + '\n'
                response['lgs_support_game_value1'] = mixed_ggw[0][0][0]
                temp.append(mixed_ggw[0][0][0])
                context['solvemixed'] += 'w = ' + str(mixed_ggw[0][0][0]) + r'\\'
                solution += 'Der Spielwert von Spieler 2 entspricht: ' + '\n'
                context['solvemixed'] += r'Der Spielwert von Spieler 2 entspricht: \\'
                solution += 'w = ' + str(mixed_ggw[0][1][0]) + '\n'
                response['lgs_support_game_value2'] = mixed_ggw[0][1][0]
                temp.append(mixed_ggw[0][1][0])
                context['solvemixed'] += 'w = ' + str(mixed_ggw[0][1][0]) + r'\\'
                responses = []
                if len(mixed_ggw[0][3][0]) or len(mixed_ggw[0][3][1]):
                    # response['results_out_of_support'].append(True)
                    solution += 'Nun müssen die erzielbaren Auszahlungen eines Spielers, durch die Wahl einer ' \
                                'Strategie, die nicht im betrachteten Support liegt ermittelt werden.\n'
                    context['solvemixed'] += r'Nun müssen die erzielbaren Auszahlungen eines Spielers, durch die ' \
                                             r'Wahl einer Strategie, die nicht im betrachteten Support liegt ' \
                                             r'ermittelt werden.\\'
                    if len(mixed_ggw[0][3][1]):
                        responses.append(True)
                        temp.append(True)
                        solution += 'Für die Auszahlung von Spieler 1 werden somit folgende Gleichungssysteme ' \
                                    'und Lösungen betrachtet: \n'
                        context['solvemixed'] += r'Für die Auszahlung von Spieler 1 werden somit folgende ' \
                                                 r'Gleichungssysteme und Lösungen betrachtet: \\'

                        game_val_mixed_1 = []
                        strats = []
                        for notsupportstrat in mixed_ggw[0][3][1]:
                            for eq in notsupportstrat[0]:
                                solution += str(eq) + ' = 0\n'
                                context['solvemixed'] += str(eq) + r' = 0\\'
                            solution += 'Dieses Gleichungssystem führt zu einer Auszahlung für Spieler 1 von:\n'
                            context['solvemixed'] += r'Dieses Gleichungssystem führt zu einer Auszahlung für ' \
                                                     r'Spieler 1 von:\\'
                            solution += 'w = ' + str(notsupportstrat[1]) + '\n'
                            context['solvemixed'] += 'w = ' + str(notsupportstrat[1]) + r'\\'
                            solution += 'Was dazu führt, dass eine Abweichung von den Support-Strategien für ' \
                                        'Spieler 1 keine Auszahlungsverbesserung hervorruft.\n'
                            context['solvemixed'] += r'Was dazu führt, dass eine Abweichung von den ' \
                                                     r'Support-Strategien für Spieler 1 keine ' \
                                                     r'Auszahlungsverbesserung hervorruft.\\'
                            game_val_mixed_1.append(notsupportstrat[1])
                            strats.append(notsupportstrat[0])
                        response['lgs_nonsupport_lgs_player1'].append(strats)
                        response['lgs_nonsupport_result_player1'].append(game_val_mixed_1)
                        temp.append(strats)
                        temp.append(game_val_mixed_1)
                    else:
                        responses.append(False)
                        temp.append(False)
                        solution += 'Für Spieler 1 liegen alle Strategien in der betrachteten Support-Menge, ' \
                                    'wodurch keine weitere Betrachtung notwendig ist.\n'
                        context['solvemixed'] += r'Für Spieler 1 liegen alle Strategien in der betrachteten ' \
                                                 r'Support-Menge, wodurch keine weitere Betrachtung notwendig ist.\\'
                        response['lgs_nonsupport_lgs_player1'].append([])
                        response['lgs_nonsupport_result_player1'].append([])
                        temp.append([])
                        temp.append([])
                    if len(mixed_ggw[0][3][0]):
                        responses.append(True)
                        temp.append(True)
                        solution += 'Für die Auszahlung von Spieler 2 werden somit folgende Gleichungssysteme ' \
                                    'und Lösungen betrachtet: \n'
                        context['solvemixed'] += r'Für die Auszahlung von Spieler 2 werden somit folgende ' \
                                                 r'Gleichungssysteme und Lösungen betrachtet: \\'

                        game_val_mixed_2 = []
                        strats = []
                        for notsupportstrat in mixed_ggw[0][3][0]:
                            for eq in notsupportstrat[0]:
                                solution += str(eq) + ' = 0\n'
                                context['solvemixed'] += str(eq) + r' = 0\\'
                            solution += 'Dieses Gleichungssystem führt zu einer Auszahlung für Spieler 2 von:\n'
                            context['solvemixed'] += r'Dieses Gleichungssystem führt zu einer Auszahlung für ' \
                                                     r'Spieler 2 von:\\'
                            solution += 'w = ' + str(notsupportstrat[1]) + '\n'
                            context['solvemixed'] += 'w = ' + str(notsupportstrat[1]) + r'\\'
                            solution += 'Was dazu führt, dass eine Abweichung von den Support-Strategien für ' \
                                        'Spieler 2 keine Auszahlungsverbesserung hervorruft.\n'
                            context['solvemixed'] += r'Was dazu führt, dass eine Abweichung von den ' \
                                                     r'Support-Strategien für Spieler 2 keine ' \
                                                     r'Auszahlungsverbesserung hervorruft.\\'
                            game_val_mixed_2.append(notsupportstrat[1])
                            strats.append(notsupportstrat[0])
                        response['lgs_nonsupport_lgs_player2'].append(strats)
                        response['lgs_nonsupport_result_player2'].append(game_val_mixed_2)
                        temp.append(strats)
                        temp.append(game_val_mixed_2)
                    else:
                        responses.append(False)
                        temp.append(False)
                        solution += 'Für Spieler 2 liegen alle Strategien in der betrachteten Support-Menge, ' \
                                    'wodurch keine weitere Betrachtung notwendig ist.\n'
                        context['solvemixed'] += r'Für Spieler 2 liegen alle Strategien in der betrachteten ' \
                                                 r'Support-Menge, wodurch keine weitere Betrachtung notwendig ist.\\'
                        response['lgs_nonsupport_lgs_player2'].append([])
                        temp.append([])
                        response['lgs_nonsupport_result_player2'].append([])
                        temp.append([])
                else:
                    solution += 'Da für beide Spieler keine Strategie nicht in der Support-Menge liegt, ' \
                                'müssen keine weiteren Berechnungen durchgeführt werden.\n'
                    context['solvemixed'] += r'Da für beide Spieler keine Strategie nicht in der Support-Menge ' \
                                             r'liegt, müssen keine weiteren Berechnungen durchgeführt werden.\\'
                    responses = [False, False]
                    temp.append(False)
                    temp.append([])
                    temp.append([])
                    temp.append(False)
                    temp.append([])
                    temp.append([])
                response['results_out_of_support'].append(responses)
                response['mixed_ggw'].append(temp)
                # for mixed_ggw in nggw2:
                #     solution += '\n'
                #     solution += str(mixed_ggw)
                #     solution += '\n'
                #     solution += str(mixed_ggw[0])
                #     solution += '\n'
                #     # Spielwert Spieler 1
                #     solution += str(mixed_ggw[0][0][0])
                #     solution += '\n'
                #     # Wahrscheinlichkeiten Spieler 1
                #     solution += str(mixed_ggw[0][0][1][0])
                #     solution += '\n'
                #     # Support Spieler 1
                #     solution += str(mixed_ggw[0][0][2])
                #     solution += '\n'
                #     # Spielwert Spieler 2
                #     solution += str(mixed_ggw[0][1][0])
                #     solution += '\n'
                #     # Wahrscheinlichkeiten Spieler 2
                #     solution += str(mixed_ggw[0][1][1][0])
                #     solution += '\n'
                #     # Support Spieler 2
                #     solution += str(mixed_ggw[0][1][2])
                #     solution += '\n'
                #     # Gleichtungssystem Spieler 1
                #     solution += str(mixed_ggw[0][2][0])
                #     solution += '\n'
                #     # Gleichtungssystem Spieler 2
                #     solution += str(mixed_ggw[0][2][1])
                #     solution += '\n'
                #     solution += str(mixed_ggw[0][3][0])
                #     solution += '\n'
                #     solution += str(mixed_ggw[0][3][1])
    sol_tex = [sol_texpure, sol_texmixed]
    return solution, sol_tex, context, response


# Spielmatrix reduzieren
def reduce_matrix(payoff_matrix_1, payoff_matrix_2=np.array([]), stop_dimension1=0, stop_dimension2=0):
    """Reduziert die Auszahlungsmatrizen solange noch strikt dominierte Strategien gefunden werden und \
    eine angegebene Mindestdimension nicht unterschritten wird

    :param payoff_matrix_1: Die Auszahlungsmatrix von Spieler 1
    :type payoff_matrix_1: ndarray
    :param payoff_matrix_2: Die Auszahlungsmatrix von Spieler 2 (default: np.array([]))
    :type payoff_matrix_2: ndarray
    :param stop_dimension1: Die Mindestanzahl der Strategien für Spieler 1, die nicht unterschritten \
    werden soll (default: 0)
    :type stop_dimension1: int
    :param stop_dimension2: Die Mindestanzahl der Strategien für Spieler 2, die nicht unterschritten \
    werden soll (default: 0)
    :type stop_dimension2: int
    :returns: Eine Liste, die die reduzierten Auszahlungsmatrizen beinhaltet, sowie eine Aussage, ob \
    überhaupt reduziert werden konnte
    :rtype: list
    """
    if not payoff_matrix_2.size:
        payoff_matrix_2 = payoff_matrix_1*-1
    # Matrizen für Spieler 1 und 2 müssen betrachtet werden
    reduced_matrix_1 = np.asarray(payoff_matrix_1)
    reduced_matrix_2 = np.asarray(payoff_matrix_2)
    shape1, shape2 = reduced_matrix_1.shape
    shape = reduced_matrix_1.shape
    dominated_strats = get_dominated_strategies(reduced_matrix_1, reduced_matrix_2)
    last_reduce = False
    reduced = False
    while dominated_strats and shape1 >= stop_dimension1 and shape2 >= stop_dimension2 and not last_reduce:
        last_reduce = True
        if shape1 - len(dominated_strats[0]) >= stop_dimension1:
            reduced_matrix_1 = np.delete(reduced_matrix_1, dominated_strats[0], axis=0)
            reduced_matrix_2 = np.delete(reduced_matrix_2, dominated_strats[0], axis=0)
        elif shape1 > stop_dimension1:
            reduced_matrix_1 = np.delete(reduced_matrix_1, dominated_strats[0][0:stop_dimension1-shape1], axis=0)
            reduced_matrix_2 = np.delete(reduced_matrix_2, dominated_strats[0][0:stop_dimension1-shape1], axis=0)
        if shape2 - len(dominated_strats[1]) >= stop_dimension2:
            reduced_matrix_1 = np.delete(reduced_matrix_1, dominated_strats[1], axis=1)
            reduced_matrix_2 = np.delete(reduced_matrix_2, dominated_strats[1], axis=1)
        elif shape2 > stop_dimension2:
            reduced_matrix_1 = np.delete(reduced_matrix_1, dominated_strats[1][0:stop_dimension2 - shape2], axis=1)
            reduced_matrix_2 = np.delete(reduced_matrix_2, dominated_strats[1][0:stop_dimension2 - shape2], axis=1)
        if reduced_matrix_1.shape < shape:
            last_reduce = False
            reduced = True
            dominated_strats = get_dominated_strategies(reduced_matrix_1, reduced_matrix_2)
            shape1, shape2 = reduced_matrix_1.shape
            shape = reduced_matrix_1.shape

    # all_compared = False
    # run = 0
    # reduced = False
    # while not all_compared:
    #     run += 1
    #     all_compared = True
    #     dimensions = reduced_matrix_1.shape
    #     reduce = []
    #     if dimensions[0] > stop_dimension1:
    #         for count in range(dimensions[0]):
    #             # reducable_line = True
    #             added = False
    #             for count_2 in range(dimensions[0]):
    #                 reducable_line = True
    #                 if count != count_2:
    #                     for count_3 in range(dimensions[1]):
    #                         if reduced_matrix_1[count][count_3] > reduced_matrix_1[count_2][count_3]
    # and reducable_line:
    #                             reducable_line = False
    #                     if reducable_line:
    #                         if not added:
    #                             reduce.append(count)
    #                             all_compared = False
    #                             added = True
    #         i = 0
    #         for count in range(len(reduce)):
    #             if dimensions[0] > stop_dimension1:
    #                 reduced_matrix_1 = np.delete(reduced_matrix_1, reduce[count] - i, 0)
    #                 reduced_matrix_2 = np.delete(reduced_matrix_2, reduce[count] - i, 0)
    #                 i += 1
    #                 reduced = True
    #                 dimensions = reduced_matrix_1.shape
    #         dimensions = reduced_matrix_1.shape
    #     reduce = []
    #
    #     if dimensions[1] > stop_dimension2:
    #         for count in range(dimensions[1]):
    #             # reducable_column = True
    #             added = False
    #             for count_2 in range(dimensions[1]):
    #                 reducable_column = True
    #                 if count != count_2:
    #                     for count_3 in range(dimensions[0]):
    #                         if reduced_matrix_2[count_3][count] > reduced_matrix_2[count_3][count_2]:
    #                             if reducable_column:
    #                                 reducable_column = False
    #                     if reducable_column and not added:
    #                         reduce.append(count)
    #                         all_compared = False
    #                         added = True
    #         i = 0
    #         for count in range(len(reduce)):
    #             if dimensions[1] > stop_dimension2:
    #                 reduced_matrix_1 = np.delete(reduced_matrix_1, reduce[count] - i, 1)
    #                 reduced_matrix_2 = np.delete(reduced_matrix_2, reduce[count] - i, 1)
    #                 i += 1
    #                 reduced = True
    #                 dimensions = reduced_matrix_1.shape
    #     if reduced_matrix_1.shape[0] <= stop_dimension1 and reduced_matrix_1.shape[1] <= stop_dimension2:
    #         all_compared = True

    return [reduced_matrix_1, reduced_matrix_2, reduced]


# Matrix aller MinMax-Strategie-Paare ausgeben
# TODO: evtl. in Game-Klasse übernehmen
def get_strategy_pairs(payoff_matrix_1, payoff_matrix_2=np.array([])):
    """Kombiniert alle Maximin-Strategien der einzelnen Spieler zu Maximin-Strategie-Kombinationen

    :param payoff_matrix_1: Die Auszahlungsmatrix von Spieler 1
    :type payoff_matrix_1: ndarray
    :param payoff_matrix_2: Die Auszahlungsmatrix von Spieler 2 (default: np.array([]))
    :type payoff_matrix_2: ndarray
    :returns: Eine Liste aller Strategie-Kombinationen durch Einsatz der Maximin-Strategien
    :rtype: product
    """
    if not payoff_matrix_2.size:
        payoff_matrix_2 = payoff_matrix_1*-1
    strategies = solve_maximin_strategies(payoff_matrix_1, payoff_matrix_2)
    strategy_pairs = product(strategies[0], strategies[1])

    return strategy_pairs


# Punkt aus unteren Spielwerten heißt Garantiepunkt
# Undominiert, wenn kein anderer Auszahlungspunkt existiert bei dem u1 und u2 >= u1* und u2*
# mode = 0 reine Strategien, mode = 1 gemischte Strategien
# TODO: Dominiertheit bei gemischten Strategien erarbeiten
def get_guaranteed_payoff(payoff_matrix_1, payoff_matrix_2=np.array([]), mode=0):
    """Berechnet den Garantiepunkt eines Spiels in reinen oder gemischten Strategien

    :param payoff_matrix_1: Die Auszahlungsmatrix von Spieler 1
    :type payoff_matrix_1: ndarray
    :param payoff_matrix_2: Die Auszahlungsmatrix von Spieler 2 (default: np.array([]))
    :type payoff_matrix_2: ndarray
    :param mode: Angabe, ob der Garantiepunkt in reinen (mode=0) oder gemischten (mode=1) Strategien \
    ermittelt werden soll (default: 0)
    :type mode: int
    :returns: Eine Liste, die den Garantiepunkt, sowie eine Aussage über Dominiertheit enthält
    :rtype: list
    """
    if not payoff_matrix_2.size:
        payoff_matrix_2 = payoff_matrix_1*-1
        #print('payoff2 generiert')
    payoff = list()
    payoff2 = set()
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
        #result = solve_using_nggw(payoff_matrix_1, payoff_matrix_2)
        result2 = get_optimal_solution(payoff_matrix_1, payoff_matrix_2)
        #print('result2')
        #print(result2[0][0])
        for solution in result2:
            #print('solution')
            #print(solution)
            #print(solution[0][0], solution[0][1])
            payoff2.add((solution[0][0][0], solution[0][1][0]))
        # print(result)
        #for players in range(len(result)):
            # print(result[players][1][-1])
        #    payoff.append(result[players][0][result[players][1][-1]])
        payoff = list(payoff2)[0]
        #print(payoff2[0][0])
        #print(payoff2[0][1])
        #payoff = payoff2

    return payoff, dominated


# Matrizen für Simplex vorbereiten
# Konstante hinzurechnen, sodass Spieler1-Matrix absolut positiv und Spieler2-Matrix absolut negativ ist
def make_matrix_ready(payoff_matrix_1, payoff_matrix_2=np.array([])):
    """Bereitet die Auszahlungsmatrizen der Spieler für die Nutzung des Simplex-Algorithmus vor

    :param payoff_matrix_1: Die Auszahlungsmatrix von Spieler 1
    :type payoff_matrix_1: ndarray
    :param payoff_matrix_2: Die Auszahlungsmatrix von Spieler 2 (default: np.array([]))
    :type payoff_matrix_2: ndarray
    :returns: Eine Liste, die beide aufbereiteten Auszahlungsmatrizen enthält
    :rtype: list
    """
    if not payoff_matrix_2.size:
        payoff_matrix_2 = payoff_matrix_1*-1
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


def use_simplex(payoff_matrix_1, payoff_matrix_2=np.array([])):
    """Ruft die Methoden zur Generierung und Lösung der linearen Probleme beider Spiele auf

    :param payoff_matrix_1: Die Auszahlungsmatrix von Spieler 1
    :type payoff_matrix_1: ndarray
    :param payoff_matrix_2: Die Auszahlungsmatrix von Spieler 2 (default: np.array([]))
    :type payoff_matrix_2: ndarray
    :returns: Die Listen mit der Lösung und allen Zwischenschritten für beide Spieler, sowie die \
    vorbereiteten Auszahlungsmatrizen
    :rtype: list
    """
    if not payoff_matrix_2.size:
        payoff_matrix_2 = payoff_matrix_1*-1
    simplex_games = make_matrix_ready(payoff_matrix_1, payoff_matrix_2)
    # print('Matrizen: ', simplex_games)
    simplex_1_solution = use_simplex_player(simplex_games[1], 0)
    simplex_2_solution = use_simplex_player(simplex_games[0], 1)

    return [simplex_1_solution, simplex_2_solution, simplex_games]


# Simplex-Verfahren für Spieler 1 anwenden
# TODO: Formatierung des Lösungswegs direkt hier machen
def use_simplex_player2(simplex_game_1):
    """Erzeugung und Lösung des linearen Problems für die Auszahlung von Spieler 2 und die von Spieler 1 \
    gespielten Wahrscheinlichkeiten

    :param simplex_game_1: Die zuvor aufbereitete Auszahlungsmatrix von Spieler 2, die nur negative Auszahlungen enthält
    :type simplex_game_1: ndarray
    :returns: Eine Liste mit der Lösung des linearen Problems, den Zwischenergebnissen und den zugeörigen \
    Pivots und den in Brüchen formatierten Zwischentableaus
    :rtype: list
    """
    #c = list()
    #game_bounds = list()
    #a = list()
    #b = list()
    #for lines in range(np.asarray(simplex_game_1).shape[0]):
    #    temp = list()
    #    for columns in range(np.asarray(simplex_game_1).shape[1]):
    #        temp.append(simplex_game_1[lines][columns])
    #    a.append(temp)
    #    b.append(1)
    a = []
    for lines in range(simplex_game_1.shape[0]):
        a.append([simplex_game_1[lines][columns] for columns in range(simplex_game_1.shape[1])])
    b = [1] * simplex_game_1.shape[0]
    #for columns in range(np.asarray(simplex_game_1).shape[1]):
    #    c.append(-1)
    #    game_bounds.append((0, None))
    c = [-1] * simplex_game_1.shape[1]
    #game_bounds = [(0, None)] * simplex_game_1.shape[1]
    # xk_arr1 = list()
    # kwargs_arr1 = list()
    solve_report_1 = SolvingSteps()
    simplex_sol = optimize.linprog(c, a, b, callback=solve_report_1.save_values)
    simplex_steps = solve_report_1.get_array_kwargs()
    simplex_steps_2 = format_solution(solve_report_1.get_array_xk())

    # print('Player 2: ', c, A, b)

    return simplex_sol, simplex_steps, simplex_steps_2


def use_simplex_player(simplex_game, player):
    """

    :param simplex_game: Auszahlungsmatrix des Spielers
    :type simplex_game: ndarray
    :param player: Spieler, dessen Wahrscheinlichkeiten ermittelt werden sollen (0 = Spieler 1 / 1 = Spieler 2)
    :type player: int
    :returns: Eine Liste mit der Lösung des linearen Problems, den Zwischenergebnissen und den zugeörigen \
    Pivots und den in Brüchen formatierten Zwischentableaus
    :rtype: list
    """
    if player == 0:
        c = [1] * simplex_game.shape[0]
        b = [-1] * simplex_game.shape[1]
        simplex_game = simplex_game.T
    else:
        c = [-1] * simplex_game.shape[1]
        b = [1] * simplex_game.shape[0]
    a = []
    for lines in range(simplex_game.shape[0]):
        a.append([simplex_game[lines][columns] for columns in range(simplex_game.shape[1])])

    solve_report = SolvingSteps()
    simplex_sol = optimize.linprog(c, a, b, callback=solve_report.save_values)
    simplex_steps = solve_report.get_array_kwargs()
    simplex_steps2 = format_solution(solve_report.get_array_xk())

    return simplex_sol, simplex_steps, simplex_steps2


# Simplex-Verfahren für Spieler 2 anwenden
# TODO: Formatierung des Lösungswegs direkt hier machen
def use_simplex_player1(simplex_game_2):
    """Erzeugung und Lösung des linearen Problems für die Auszahlung von Spieler 1 und die von Spieler 2 \
    gespielten Wahrscheinlichkeiten

    :param simplex_game_2: Die zuvor aufbereitete Auszahlungsmatrix von Spieler 1, die nur positive \
    Auszahlungen enthält
    :type simplex_game_2: ndarray
    :returns: Eine Liste mit der Lösung des linearen Problems, den Zwischenergebnissen und den zugeörigen \
    Pivots und den in Brüchen formatierten Zwischentableaus
    :rtype: list
    """
    #c = list()
    #game_bounds = list()
    #a = list()
    #b = list()
    #for lines in range(np.asarray(simplex_game_2).shape[0]):
    #    c.append(1)
    #    game_bounds.append((0, None))
    c = [1] * simplex_game_2.shape[0]
    #game_bounds = [(0, None)] * simplex_game_2.shape[0]
    #for columns in range(np.asarray(simplex_game_2).shape[1]):
    #    #b.append(-1)
    #    temp = list()
    #    for lines in range(np.asarray(simplex_game_2).shape[0]):
    #        temp.append(simplex_game_2[lines][columns])
    #    a.append(temp)
    a = []
    for columns in range(simplex_game_2.shape[1]):
        a.append([simplex_game_2[lines][columns] for lines in range(simplex_game_2.shape[0])])
    b = [-1] * simplex_game_2.shape[1]

    # xk_arr2 = list()
    # kwargs_arr2 = list()
    solve_report2 = SolvingSteps()
    simplex_sol = optimize.linprog(c, a, b, callback=solve_report2.save_values)
    simplex_steps = solve_report2.get_array_kwargs()
    simplex_steps_2 = format_solution(solve_report2.get_array_xk())

    # print('Player 1: ', c, A, b)

    return simplex_sol, simplex_steps, simplex_steps_2


def get_dominated_strategies(payoff_matrix_1, payoff_matrix_2=np.array([])):
    """Ermittelt alle strikt dominierten Strategien eines Zwei-Personenspiels

    :param payoff_matrix_1: Die Auszahlungsmatrix von Spieler 1
    :type payoff_matrix_1: ndarray
    :param payoff_matrix_2: Die Auszahlungsmatrix von Spieler 2 (default: np.array([]))
    :type payoff_matrix_2: ndarray
    :returns: Eine Liste der für den jeweiligen Spieler strikt dominierten Strategien
    :rtype: list
    """
    if not payoff_matrix_2.size:
        payoff_matrix_2 = payoff_matrix_1*-1
    shape1, shape2 = payoff_matrix_1.shape
    dominated_player_1 = set()
    dominated_player_2 = set()
    for lincol in combinations(range(shape1), 2):
        if np.all(payoff_matrix_1[lincol[0], :] > payoff_matrix_1[lincol[1], :]):
            dominated_player_1.add(lincol[1])
        elif np.all(payoff_matrix_1[lincol[0], :] < payoff_matrix_1[lincol[1], :]):
            dominated_player_1.add(lincol[0])
    #for line in range(shape1):
        #print('Zeile')
        #print(line)
        #for compared_line in (lin for lin in range(shape1) if lin != line):
            #dominated_1 = True
            #for column in range(shape2):
                #if payoff_matrix_1[line][column] <= payoff_matrix_1[compared_line][column]:
                    #dominated_1 = False
                    #break
            #if dominated_1:
                #dominated_player_1.add(compared_line)
            #print(compared_line, dominated_1)
    for collin in combinations(range(shape2), 2):
        if np.all(payoff_matrix_2[:, collin[0]] > payoff_matrix_2[:, collin[1]]):
            dominated_player_2.add(collin[1])
        elif np.all(payoff_matrix_2[:, collin[0]] < payoff_matrix_2[:, collin[1]]):
            dominated_player_2.add(collin[0])
    #for column in range(shape2):
    #    for compared_column in (col for col in range(shape2) if col != column):
    #        dominated_2 = True
    #        for line in range(shape1):
    #            if payoff_matrix_2[line][column] <= payoff_matrix_2[line][compared_column]:
    #                dominated_2 = False
    #                break
    #        if dominated_2:
    #            dominated_player_2.add(compared_column)
    return list(dominated_player_1), list(dominated_player_2)


def get_possible_solutions(payoff_matrix_1, payoff_matrix_2=np.array([]), sym=np.array([]), funcs=np.array([])):
    """Ermittelt mögliche, auf die Einhaltung der Bedingungen eines Nash-Gleichgewichts zu überprüfende, \
    Support-Kombinationen beider Spieler in gemischten Strategien

    :param payoff_matrix_1: Die Auszahlungsmatrix von Spieler 1
    :type payoff_matrix_1: ndarray
    :param payoff_matrix_2: Die Auszahlungsmatrix von Spieler 2 (default: np.array([]))
    :type payoff_matrix_2: ndarray
    :param sym: Die Variablen der Gleichungssysteme (default: np.array([]))
    :type sym: ndarray
    :param funcs: Eine Liste der Gleichungssysteme beider Spieler (default: np.array([]))
    :type funcs: ndarray
    :returns: Eine Liste aller vorselektieren Support-Kombinationen, die auf Existenz eines Nash-Gleichgewichts \
    überpüft werden müssen
    :rtype: list
    """
    if not payoff_matrix_2.size:
        payoff_matrix_2 = payoff_matrix_1*-1
    shape1, shape2 = payoff_matrix_1.shape
    possible_sols = []
    if not sym.size:
        sym = generate_symbols(payoff_matrix_1)
    if not funcs.size:
        funcs = generate_functions(payoff_matrix_1, payoff_matrix_2)
    #print(funcs)
    dominated_strategies1, dominated_strategies2 = get_dominated_strategies(payoff_matrix_1, payoff_matrix_2)
    #print(dominated_strategies1, dominated_strategies2)
    #print(sym[0][0])
    for supp2 in (s for s in powerset(shape2) if len(s) > 0 and not any(x in s for x in dominated_strategies2)):
        for supp1 in (s for s in powerset(shape1) if len(s) > 0 and not any(x in s for x in dominated_strategies1)):
            temp = []
            temp2 = []
            #temp_eq = 0
            #temp_eq2 = 0
            temp_eq = funcs[1][-1]
            temp_eq2 = funcs[0][-1]
            #print(supp1, supp2)
            #for strat2 in sym[1][0]:
            #    temp_eq2 += strat2
            #for strat1 in sym[0][0]:
            #    temp_eq += strat1
            #temp_eq -= 1
            #temp_eq2 -= 1
            strats1 = np.array((temp_eq))
            strats2 = np.array((temp_eq2))
            #print(supp1, supp2)
            for strat2 in supp2:
                strats1 = np.append(strats1, funcs[1][strat2])
            for strat1 in supp1:
                strats2 = np.append(strats2, funcs[0][strat1])
            for i in range(len(sym[0][0])):
                if i not in supp1:
                    strats1 = np.append(strats1, sym[0][0][i])
            for i in range(len(sym[1][0])):
                if i not in supp2:
                    strats2 = np.append(strats2, sym[1][0][i])
            solu = solve(strats1, dict=True, check=False, force=True)
            solu2 = solve(strats2, dict=True, check=False, force=True)
            #print(solu, solu2)
            solution = True
            temp.append([strats1])
            temp2.append([strats2])
            if solu and solu2:
                try:
                    for key in solu[0]:
                        if solu[0][key] < 0 and key in sym[0][0]:
                            #print(solu[0][key])
                            solution = False
                    for key in solu2[0]:
                        if solu2[0][key] < 0 and key in sym[1][0]:
                            #print(solu2[0][key])
                            solution = False
                except TypeError:
                    solution = False
            if solution:
                #temp = np.append(temp, np.array((solu)))
                #temp = [[strats1], solu, [deepcopy(supp1), deepcopy(supp2)]]
                temp.append(solu)
                temp.append([deepcopy(supp1), deepcopy(supp2)])
                #temp = np.append(temp, np.array((deepcopy(supp1), deepcopy(supp2))))
                #temp2 = np.append(temp2, np.array((solu2)))
                #temp2 = np.append(temp2, np.array((deepcopy(supp1), deepcopy(supp2))))
                temp2.append(solu2)
                temp2.append([deepcopy(supp1), deepcopy(supp2)])
                combined = np.array((temp, temp2))
                possible_sols.append(combined)
    return possible_sols


def get_optimal_solution(payoff_matrix_1, payoff_matrix_2=np.array([])):
    """Überprüft mögliche Support-Kombinationen auf die Einhaltung der Bedingungen eines Nash-Gleichgewichts

    :param payoff_matrix_1: Die Auszahlungsmatrix von Spieler 1
    :type payoff_matrix_1: ndarray
    :param payoff_matrix_2: Die Auszahlungsmatrix von Spieler 2 (default: np.array([}]))
    :type payoff_matrix_2: ndarray
    :returns: Eine Liste aller Nash-Gleichgewichte, inklusive deren Supports, Auszahlungen und \
    Wahrscheinlichkeitsverteilungen über die reinen Strategien oder False falls kein Nash-Gleichgewicht \
    ermittelt werden konnte.
    :rtype: list
    """
    if not payoff_matrix_2.size:
        payoff_matrix_2 = payoff_matrix_1*-1
    #payoff_matrix_1, payoff_matrix_2, reduced = reduce_matrix(payoff_matrix_1, payoff_matrix_2)
    #print(payoff_matrix_1)
    #print(payoff_matrix_2)
    shape1, shape2 = payoff_matrix_1.shape
    values_ret = []
    sym = generate_symbols(payoff_matrix_1)
    funcs = generate_functions(payoff_matrix_1, payoff_matrix_2, symbs=sym)
    pos_solutions = get_possible_solutions(payoff_matrix_1, payoff_matrix_2, funcs=funcs, sym=sym)
    #print(pos_solutions)
    #print(pos_solutions)
    for sol in pos_solutions:
        #print('sol')
        #print(sol)
        #temp_funcs = []
        #temp_funcs2 = []
        #for i in range(shape2):
        #    if i not in sol[0][2][1]:
        #        temp_funcs.append(funcs[1][0][i])
        #for i in range(shape1):
        #    if i not in sol[1][2][0]:
        #        temp_funcs2.append(funcs[0][0][i])
        temp_funcs = np.array(funcs[1][-1])
        temp_funcs2 = np.array(funcs[0][-1])
        #print(sol[0], sol[1])
        #temp_funcs.append(funcs[1][0][-1])
        #temp_funcs2.append(funcs[0][0][-1])
        #print('sol01, sol11')
        #print(sol[0][1], sol[1][1])
        if len(sol[0][1]) > 0 and len(sol[1][1]) > 0:
            #print(sym[0][0])
            #sup1 = []
            for symb in sym[0][0]:
                #print('1. Test')
                #print(sol[0][1])
                #print(sol[0][1][0][symb])
                try:
                    #print(sol[0][1][0])
                    #print(symb - sol[0][1][0][symb])
                    #print(sol[0])
                    temp_funcs = np.append(temp_funcs, symb - sol[0][1][0][symb])
                    #temp_funcs.append(symb - sol[0][1][0][symb])
                    #if sol[0][1][0][symb] != 0:
                    #    sup1.append(sym[0][0].index(symb))
                except KeyError:
                    pass
            #sol[0][2][0] = deepcopy(sup1)
            #sol[1][2][0] = deepcopy(sup1)
                #print(temp_funcs)
           # print(sym[1][0])
            #del sup1[:]
            #sup2 = []
            for symb in sym[1][0]:
                #print('2. Test')
                #print(sol[1][1])
                #print(sol[1][1][0][symb])
                try:
                    #print(sol[1][1][0])
                    #print(symb - sol[1][1][0][symb])
                    temp_funcs2 = np.append(temp_funcs2, symb - sol[1][1][0][symb])
                    #temp_funcs2.append(symb - sol[1][1][0][symb])
                    #print(sym[1][0])
                    #if sol[1][1][0][symb] != 0:
                    #    sup2.append(sym[1][0].index(symb))
                except KeyError:
                    pass
                #print(temp_funcs2)
            #sol[1][2][1] = deepcopy(sup2)
            #sol[0][2][1] = deepcopy(sup2)
            #del sup2[:]
            #print('Ende Vorbereitung')
            vals1 = []
            vals2 = []
            #vals1 = []
            #vals2 = []
            for i in range(shape2):
                #print('support:')
                #print(i, sol[0][2][1])
                if i not in sol[0][2][1]:
                    try:
                        tempo = np.append(temp_funcs, funcs[1][i])
                        #tempo = deepcopy(temp_funcs)
                        #tempo.append(funcs[1][0][i])
                        #print(tempo)
                        #print('tempo:')
                        #print(tempo)
                        tempo_sol = solve(tempo, dict=True, check=False, force=True)
                        vals1.append(np.array([tempo, tempo_sol[0][sym[1][1]]]))
                        #print(tempo_sol[0][sym[1][1]])
                        #print(vals1)
                        #print('vals1:')
                        #print(vals1)
                        #vals1.append([tempo, tempo_sol[0][sym[1][1]]])
                        #print(tempo_sol)
                    except KeyError:
                        pass
                    except IndexError:
                        pass
            for i in range(shape1):
                #print('support:')
                #print(i, sol[1][2][0])
                if i not in sol[1][2][0]:
                    try:
                        tempo = np.append(temp_funcs2, funcs[0][i])
                        #tempo = deepcopy(temp_funcs2)
                        #tempo.append(funcs[0][0][i])
                        #print(tempo)
                        #print('tempo:')
                        #print(tempo)
                        tempo_sol = solve(tempo, dict=True, check=False, force=True)
                        vals2.append(np.array([tempo, tempo_sol[0][sym[0][1]]]))
                        #print(tempo_sol[0][sym[0][1]])
                        #print(vals2)
                        #print('vals2:')
                        #print(vals2)
                        #vals2.append([tempo, tempo_sol[0][sym[0][1]]])
                        #print(tempo_sol)
                    except KeyError:
                        pass
                    except IndexError:
                        pass
            #print(vals1[:, 1], vals2[:, 1])
            vals1 = np.asarray(vals1)
            vals2 = np.asarray(vals2)
            #print(vals1)
            #print(vals2)
            #other_sol = solve(temp_funcs, dict=True, check=False, force=True)
            #other_sol2 = solve(temp_funcs2, dict=True, check=False, force=True)
            optimal = False
            #try:
            #print('lensol01, lenvals1')
            #print(len(sol[0][1]), len(vals1))
            #print(sol[0][1][0], vals1)
            #print('lensol11, lenvals2')
            #print(len(sol[1][1]),len(vals2))
            #print(sol[1][1][0], vals2)
            optimal1 = False
            optimal2 = False
            #if (len(sol[0][1]) > 0 and len(vals1) > 0) or (len(sol[1][1]) > 0 and len(vals2) > 0):
            #    # print(sol[0][1][0][sym[0][1]], vals1[:, 1])
            #    # print(sol[1][1][0][sym[1][1]], vals2[:, 1])
            #    print('other maxima')
            #    print(max(vals2[:, 1]), max(vals1[:, 1]))
            #    print(sol[0][1][0][sym[0][1]], sol[1][1][0][sym[1][1]])
            #    if sol[0][1][0][sym[0][1]] >= max(vals2[:, 1]) and sol[1][1][0][sym[1][1]] >= max(vals1[:, 1]):
            #        optimal = True
            if len(sol[0][1]) > 0 and len(vals1) > 0:
                #print('other maxima')
                #print(max(vals1[:,1]))
                if sol[0][1][0][sym[0][1]] >= max(vals1[:, 1]):
                    optimal1 = True
            if len(sol[1][1]) > 0 and len(vals2) > 0:
                #print('other maxima')
                #print(max(vals2[:, 1]))
                if sol[1][1][0][sym[1][1]] >= max(vals2[:, 1]):
                    optimal2 = True
            if len(vals2) == 0:
                optimal2 = True
            if len(vals1) == 0:
                optimal1 = True
            optimal = optimal1 and optimal2
            #except IndexError:
            #    optimal = False
            #print(sol[0][0], sol[0][1])
            #print(sol[1][0], sol[1][1])
            #print(sol[0][2], sol[1][2])
            #print(sym[0][1])
            #print(vals1, vals2)
            #try:
                #print(vals1[:, 1], sol[0][1][0][sym[0][1]])
                #print(vals2[:, 1], sol[1][1][0][sym[1][1]])
            #except IndexError:
            #    pass
            #except KeyError:
            #    pass
            #print(vals1[:, 0])
            #print(vals2[:, 0])
            if optimal:
                # print(sol[0][0], sol[0][1])
                # print(sol[1][0], sol[1][1])
                # print(sol[0][2], sol[1][2])
                # print(other_sol[0][sym[0][1]], sol[0][1][0][sym[0][1]])
                # print(other_sol2[0][sym[1][1]], sol[1][1][0][sym[1][1]])
                # print(temp_funcs)
                # print(temp_funcs2)
                # Spielwert Spieler 1, Spielwert Spieler 2
                # values.append([sol[1][1][0][sym[1][1]], sol[0][1][0][sym[0][1]]])
                # Strategiewahrscheinlichkeiten Spieler 1
                # values.append([sol[0][0], sol[0][1]])
                # Strategiewahrscheinlichkeiten Spieler 2
                # values.append([sol[1][0], sol[1][1]])
                # gewählter Support Spieler 1
                # values.append([sol[0][2][0]])
                # gewählter Support Spieler 2
                # values.append([sol[0][2][1]])
                # zu prüfende Strategien nicht im Support, bzgl. Spielwert Spieler 2:
                # values.append([temp_funcs, other_sol[0][sym[0][1]]])
                # zu prüfende Strategien nicht im Support, bzgl. Spielwert Spieler 1:
                # values.append([temp_funcs2, other_sol2[0][sym[1][1]]])
                # Spielwert Spieler 1, gewählte Strategien Spieler 1, gewählter Support Spieler 1
                #values = np.array([sol[1][1][0][sym[1][1]], sol[0][1], sol[0][2][0]])
                values = [[sol[1][1][0][sym[1][1]], sol[0][1], sol[0][2][0]],
                          [sol[0][1][0][sym[0][1]], sol[1][1], sol[1][2][1]], [sol[0][0], sol[1][0]], [vals1, vals2]]
                # Spielwert Spieler 2, gewählte Strategien Spieler 2, gewählter Support Spieler 2
                #values = np.append(values, [sol[0][1][0][sym[0][1]], sol[1][1], sol[1][2][1]])
                # Betrachtetes LGS für Wahrscheinlichkeiten Spieler 1 / 2
                #values = np.append(values, [sol[0][0], sol[1][0]])
                # Zu betrachtendes LGS nicht im Support für Spieler 1 / 2 + Ergebnis
                #values = np.append(values, [vals1, vals2])

                #print('Optimum gefunden')
                #print(values)

                values_ret.append([values])

    if len(values_ret) > 0:
        return values_ret
    else:
        return []


def generate_symbols(payoff_matrix):
    """Erzeugt die für die Lösung eines Gleichungssystems benötigten Variablen

    :param payoff_matrix: Die Auszahlungsmatrix von Spieler 1
    :returns: Ein NumPy-Array, das für beide Spieler alle Variablen der Wahrscheinlichkeitsverteilung und \
    des Spielwerts beinhaltet
    :rtype: ndarray
    """
    symbols_player_1 = symbols('p0:%d' % payoff_matrix.shape[0], nonnegative=True, seq=True)
    symbols_player_2 = symbols('q0:%d' % payoff_matrix.shape[1], nonnegative=True, seq=True)
    w = symbols('w')

    return np.array((np.array((symbols_player_1, w)), np.array((symbols_player_2, w))))


def generate_functions(payoff_matrix_1, payoff_matrix_2=np.array([]), symbs=np.array([])):
    """Erzeugt ein zu lösendes Gleichungssystem

    :param payoff_matrix_1: Die Auszahlungsmatrix von Spieler 1
    :type payoff_matrix_1: ndarray
    :param payoff_matrix_2: Die Auszahlungsmatrix von Spieler 2 (default: np.array([]))
    :type payoff_matrix_2: ndarray
    :param symbs: Die Variablen des Gleichungssystems (default: np.array([]))
    :type symbs: ndarray
    :returns: Ein NumPy-Array, das die Gleichungssysteme der beiden Spieler beinhaltet
    :rtype: ndarray
    """
    if not payoff_matrix_2.size:
        payoff_matrix_2 = payoff_matrix_1*-1
    if not symbs.size:
        symbs = generate_symbols(payoff_matrix_1)
    payoff_matrix_2 = payoff_matrix_2.transpose()
    eqs1 = []
    eqs2 = []
    # Wahrscheinlichkeiten für Spieler 2 und Auszahlung für Spieler 1
    temp2 = 0
    temp3 = 0
    for i in range(payoff_matrix_1.shape[0]):
        temp = 0
        for j in range(payoff_matrix_1.shape[1]):
            temp += payoff_matrix_1[i][j]*symbs[1][0][j]
        temp -= 1*symbs[1][1]
        eqs1 = np.append(eqs1, temp)
        #temp2 += 1*symbs[1][0][i]
        temp3 += 1*symbs[0][0][i]
    temp3 -= 1
    #eqs1 = np.append(eqs1, temp2)

    # Wahrscheinlichkeiten für Spieler 1 und Auszahlung für Spieler 2
    for i in range(payoff_matrix_2.shape[0]):
        temp = 0
        #print(payoff_matrix_2)
        for j in range(payoff_matrix_2.shape[1]):
            temp += payoff_matrix_2[i][j]*symbs[0][0][j]
        temp -= 1*symbs[0][1]
        eqs2 = np.append(eqs2, temp)
        #temp2 += 1*symbs[0][0][i]
        temp2 += 1 * symbs[1][0][i]
    temp2 -= 1
    eqs1 = np.append(eqs1, temp2)
    eqs2 = np.append(eqs2, temp3)
    return np.array((np.array(eqs1), np.array(eqs2)))


# Lösung mit Bedingungen für NGGW
# Gemischte Maximin-Strategien der Spieler
def solve_using_nggw(payoff_matrix_1, payoff_matrix_2=np.array([])):
    """Ursprüngliche Methode zur Lösung allgemeiner gemischten Erweiterungen, jedoch durch \
    get_optimal_solution ersetzt

    :param payoff_matrix_1: Die Auszahlungsmatrix von Spieler 1
    :type payoff_matrix_1: ndarray
    :param payoff_matrix_2: Die Auszahlungsmatrix von Spieler 2
    :type payoff_matrix_2: ndarray
    :returns: Eine Liste, die für jeden Spieler die Lösung der Gleichungssysteme, die Variablen und \
    die Gleichungssysteme beinhaltet
    :rtype: list
    """
    if not payoff_matrix_2.size:
        payoff_matrix_2 = payoff_matrix_1*-1
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
    """Formatiert eine Lösungsliste in Bruchschreibweise um.

    :param solution_array: Die neu zu formatierende Liste
    :type solution_array: list
    :returns: Eine Liste bestehend aus Einträgen in der Bruchschreibweise
    :rtype: list
    """
    # print('Formatting: ')
    #solution = list()
    # print(np.asarray(solution_array))
    # print('Formatting:')
    # print(np.asarray(solution_array))
    solution = []
    for line in range(np.asarray(solution_array).shape[0]):
        if np.amin(solution_array[line]) or np.amax(solution_array[line]) != 0:
            solution.append([nsimplify(solution_array[line][column], tolerance=0.0001, rational=True) for column in
                             range(np.asarray(solution_array).shape[1])])
    #for line in range(np.asarray(solution_array).shape[0]):
    #    temp = list()
    #    for column in range(np.asarray(solution_array).shape[1]):
    #        temp.append(nsimplify(solution_array[line][column], tolerance=0.0001, rational=True))
    #        # print(nsimplify(solution_array[line][column], tolerance=0.0001, rational=True))
    #    if np.amin(temp) != 0 or np.amax(temp) != 0:
    #        solution.append(temp)
    # print(solution)

    return solution


# Callable Methode um Zwischenschritte des Simplex abzufangen
class SolvingSteps:
    """Macht die Zwischentableaus des Simplex-Algorithmus auswertbar


    """
    def __init__(self):
        """Konstruktor, der jedem Objekt der Klasse die Instanzvariablen initiiert

        """
        self.__array_xk = []
        self.__array_kwargs = []
        # print('Initialisiert mit:')
        # print('xk:')
        # print(self.__array_xk)
        # print('kwargs:')
        # print(self.__array_kwargs)

    def save_values(self, xk, **kwargs):
        """Methode um die an die übergebenen Parameter in den Instanzvariablen zu speichern

        :param xk: Zwischentableau des Simplex-Algorithmus
        :type xk: list
        :param kwargs: Dictionary mit zusätzlichen Informationen zum Simplex-Zwischenschritt
        :type kwargs: dict
        :returns:
        """
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
        """

        :returns: Die als Listen-Instanzvariable gespeicherten Dictionairies
        :rtype: list
        """
        return self.__array_kwargs

    # Funktion um Parameter des Simplex abzufragen
    def get_array_xk(self):
        """

        :returns: Die als Listen-Instanzvariable gespeicherten Zwischentableaus
        :rtype: list
        """
        return self.__array_xk
