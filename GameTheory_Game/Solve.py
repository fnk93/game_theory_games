import numpy as np
from scipy import optimize
from sympy.solvers import solve
from sympy import nsimplify, symbols, Eq

# TODO: Lösungstableaus auswerten und darstellen
# TODO: Möglichkeit mehrere Lösungswege des Simplex abzubilden finden und abbilden
# TODO: Daten für Lösungsweg sammeln lassen und an neue Klasse zur Darstellung
# TODO: bzw. zum Export in LaTeX oder PDF übergeben


class Solve(object):

    def __init__(self, game):
        self.__matrix = game.get_matrix()
        self.__matrix2 = game.get_matrix2()
        self.__determined = False
        self.__determinedIntervall = []
        self.__max_value_player1 = 0
        self.__min_value_player1 = 0
        self.__max_value_player2 = 0
        self.__min_value_player2 = 0
        self.is_determined()
        self.__maximin_strategies1 = []
        self.__maximin_strategies2 = []
        self.solve_strategies()
        self.__reduced_matrix = []
        self.__solutions = []
        self.solving_array()
        self.__reduced = False
        self.reduce_matrix()
        self.__simplex_game = []
        self.__added_constant = 0
        self.make_array_ready()
        self.__c1 = []
        self.__A1 = []
        self.__b1 = []
        self.__game_bounds1 = []
        self.__simplex1 = ''
        self.__simplex1_solving = []
        self.__simplex1_solving_xk = []
        self.use_simplex1()
        self.__c2 = []
        self.__A2 = []
        self.__b2 = []
        self.__game_bounds2 = []
        self.__simplex2 = ''
        self.__simplex2_solving = []
        self.__simplex2_solving_xk = []
        self.use_simplex2()


    # Determiniertheit des Spiels bestimmen
    # Determiniertheitsintervall berechnen
    # In reinen Strategien
    def is_determined(self):
        # Oberer Spielwert für Spieler 1 und 2
        max_player1 = []
        for count in range(self.__matrix.transpose().shape[0]):
            max_player1.append(max(self.__matrix.transpose()[count]))
        self.__max_value_player1 = min(max_player1)
        max_player2 = []
        for count in range(self.__matrix2.shape[0]):
            max_player2.append(max(self.__matrix2[count]))
        self.__max_value_player2 = min(max_player2)

        # Unterer Spielwert für Spieler 1 und 2
        min_player1 = []
        for count in range(self.__matrix.shape[0]):
            min_player1.append(min(self.__matrix[count]))
        self.__min_value_player1 = max(min_player1)
        min_player2 = []
        for count in range(self.__matrix2.transpose().shape[0]):
            min_player2.append(min(self.__matrix2.transpose()[count]))
        self.__min_value_player2 = max(min_player2)

        if self.__min_value_player1 == self.__max_value_player1:
            self.__determined = True
        else:
            self.__determined = False
        self.__determinedIntervall.append(self.__min_value_player1)
        self.__determinedIntervall.append(self.__max_value_player1)

    # Ermittlung der Maximin-Strategien beider Spieler
    def solve_strategies(self):
        for count in range(self.__matrix.shape[0]):
            if min(self.__matrix[count]) == self.__min_value_player1:
                self.__maximin_strategies1.append(count+1)

        for count in range(self.__matrix.transpose().shape[0]):
            if max(self.__matrix.transpose()[count]) == self.__max_value_player1:
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
        return self.__min_value_player1

    # Oberer Spielwert
    def get_high_value(self):
        return self.__max_value_player1

    # Funktion um Ergebnisse in Textform zu präsentieren
    def output(self):
        if self.__determined:
            print('Spiel ist determiniert mit Spielwert: ' + str(self.__min_value_player1))
            print('Auszahlung für Spieler 1 in reinen Strategien: ' + str(self.__min_value_player1))
            print('Auszahlung für Spieler 2 in reinen Strategien: ' + str(-1 * self.__max_value_player1))
        else:
            print('Spiel ist nicht determiniert mit Indeterminiertheitsintervall: ' + str(self.__determinedIntervall))
            for count in range(len(self.__solutions)):
                print('Auszahlung für Spieler 1 in reinen Strategien: ' +
                      str(self.__matrix[self.__solutions[count][0]-1][self.__solutions[count][1]-1]))
                print('Auszahlung für Spieler 2 in reinen Strategien: ' +
                      str(-1 * self.__matrix[self.__solutions[count][0]-1][self.__solutions[count][1]-1]))
        print('Maximin-Strategie(n) von Spieler 1: ' + str(self.__maximin_strategies1))
        print('Maximin-Strategie(n) von Spieler 2: ' + str(self.__maximin_strategies2))
        print('Strategiekombinationen: ' + str(self.__solutions))
        if self.__reduced:
            print('Reduziertes Spiel: \n' + str(self.__reduced_matrix))
        print('Spielwert für Spieler 1 in gemischten Strategien: ' +
              str(nsimplify((1/((self.__simplex1.fun))) + (self.__added_constant - 1), tolerance=0.0001, rational=True)))
        for count in range(len(self.__simplex1.x)):
            print('Wahrscheinlichkeit Strategie ' + str(count+1) + ' für Spieler 1: ' +
                  str(nsimplify((self.__simplex1.x[count]) * ((1/(self.__simplex1.fun))), tolerance=0.0001, rational=True)))
        # print(self.__simplex2)
        for count in range(len(self.__simplex2.x)):
            print('Wahrscheinlichkeit Strategie ' + str(count+1) + ' für Spieler 2: ' +
                  str(nsimplify(abs((self.__simplex2.x[count]) * (1 / (self.__simplex2.fun))), tolerance=0.0001, rational=True)))
        #print(self.__simplex1_solving)
        self.nggw()
        print(self.__simplex1)
        print(self.__simplex2)

    # Matrix reduzieren falls möglich
    def reduce_matrix(self):
        reduced_matrix = np.asarray(self.__matrix)
        all_compared = False
        run = 0
        while not all_compared:
            run += 1
            all_compared = True
            dimensions = reduced_matrix.shape
            reduce = []
            if reduced_matrix.shape[0] > 2:
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
                dimensions = reduced_matrix.shape
            reduce = []

            if reduced_matrix.shape[1] > 2:
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
            if reduced_matrix.shape[0] <= 2 and reduced_matrix.shape[1] <= 2:
                all_compared = True
        self.__reduced_matrix = reduced_matrix


    # Lösungsarray der Minimax-Strategien
    def solving_array(self):
        for count in range(np.asarray(self.__maximin_strategies1).shape[0]):
            for count_2 in range(np.asarray(self.__maximin_strategies2).shape[0]):
                self.__solutions.append([self.__maximin_strategies1[count], self.__maximin_strategies2[count_2]])

    # Array für Simplex vorbereiten (keine negativen Auszahlungen)
    def make_array_ready(self):
        self.__added_constant = np.amin(self.__matrix)
        if self.__added_constant < 1:
            self.__simplex_game = self.__matrix - (self.__added_constant - 1)
        else:
            self.__simplex_game = self.__matrix

    # Simplex Algorithmus für Spieler 1 nutzen
    # Zwischenergebnisse abfangen und speichern
    def use_simplex1(self):
        for count_lin in range(np.asarray(self.__simplex_game).shape[0]):
            self.__c1.append(1)

        for count_col in range(np.asarray(self.__simplex_game).shape[1]):
            temp = []
            for count_lin in range(np.asarray(self.__simplex_game).shape[0]):
                temp.append(self.__simplex_game[count_lin][count_col] * -1)
            self.__A1.append(temp)

        for count_col in range(np.asarray(self.__simplex_game).shape[1]):
            self.__b1.append(-1)

        for count_lin in range(np.asarray(self.__simplex_game).shape[0]):
            self.__game_bounds1.append((0, None))

        # TODO: optimize.linprog_verbose_callback evtl. nutzen
        temp_solver = SolvingSteps()
        self.__simplex1 = optimize.linprog(self.__c1, self.__A1, self.__b1,
                                           callback=temp_solver)
        self.__simplex1_solving = temp_solver.getArrayKwargs()
        self.__simplex1_solving_xk = temp_solver.getArrayXk()

    # Simplex Algorithmus für Spieler 2 nutzen
    # Zwischenergebnisse abfangen und speichern
    def use_simplex2(self):
        for count_col in range(np.asarray(self.__simplex_game).shape[1]):
            self.__c2.append(-1)

        for count_lin in range(np.asarray(self.__simplex_game).shape[0]):
            temp = []
            for count_col in range(np.asarray(self.__simplex_game).shape[1]):
                temp.append(self.__simplex_game[count_lin][count_col])
            self.__A2.append(temp)

        for count_lin in range(np.asarray(self.__simplex_game).shape[0]):
            self.__b2.append(1)

        for count_col in range(np.asarray(self.__simplex_game).shape[1]):
            self.__game_bounds2.append((0, None))

        # TODO: optimize.linprog_verbose_callback evtl. nutzen
        temp_solver = SolvingSteps()
        self.__simplex2 = optimize.linprog(self.__c2, self.__A2, self.__b2,
                                           callback=temp_solver)
        self.__simplex2_solving = temp_solver.getArrayKwargs()
        self.__simplex2_solving_xk = temp_solver.getArrayXk()

    # Lösung mit Nash-GGW Bedingung
    def nggw(self):
        # Gemischte Strategien p für Spieler 1 und Spielwert für Spieler 2
        p = symbols('p:'+str(self.__reduced_matrix.shape[0]), nonnegative=True)
        w = symbols('w', real=True)
        #p = []
        u = []
        #w = Symbol('w', real=True)
        #for count in range(self.__reduced_matrix.shape[0]):
        #    tempo = Symbol("p" + str(count+1), nonnegative=True)
        #    p.append(tempo)
        #    print(tempo.assumptions0)
        #p.append(w)
        for count in range(self.__reduced_matrix.shape[1]):
            temp = 0
            for count_2 in range(self.__reduced_matrix.shape[0]):
                temp += (self.__reduced_matrix[count_2][count]*-1*p[count_2])
            #temp += (p[len(p)-1])*-1
            u.append(Eq(temp, w))
        temp2 = 0
        for count in range(len(p)):
            temp2 += 1*p[count]
        #temp2 -= 1
        u.append(Eq(temp2, 1))
        print('P:')
        print(p)
        print('U1:')
        print(u)
        print(solve(u))
        ngg1 = solve(u)

        # Gemischte Strategien q für Spieler 2 und Spielwert für Spieler 1
        q = symbols('q:'+str(self.__reduced_matrix.shape[1]), nonnegative=True)
        #q = []
        u2 = []
        w2 = symbols('w2', real=True)
        #for count in range(self.__reduced_matrix.shape[1]):
        #    tempo = Symbol("q" + str(count+1), nonnegative=True)
        #    q.append(tempo)
        #    print(tempo.assumptions0)
        #q.append(w2)
        for count in range(self.__reduced_matrix.shape[0]):
            temp = 0
            for count_2 in range(self.__reduced_matrix.shape[1]):
                temp += (self.__reduced_matrix[count][count_2]*q[count_2])
            #temp += (q[len(q)-1])*-1
            u2.append(Eq(temp, w2))
        temp2 = 0
        for count in range(len(q)):
            temp2 += 1*q[count]
        #temp2 -= 1
        u2.append(Eq(temp2, 1))
        print('Q:')
        print(q)
        print('U2:')
        print(u2)
        print(solve(u2))
        ngg2 = solve(u2)



# Callable Methode um Zwischenschritte des Simplex abzufangen
class SolvingSteps():

    def __init__(self):
        self.__array_xk = []
        self.__array_kwargs = []

    def __call__(self, xk, **kwargs):
        self.__array_xk.append(xk)
        self.__array_kwargs.append(kwargs['tableau'])

    def getArrayKwargs(self):
        return self.__array_kwargs

    def getArrayXk(self):
        return self.__array_xk
