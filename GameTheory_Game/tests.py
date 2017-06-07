import numpy as np
from GameTheory_Game.Solving_Methods import *
from GameTheory_Game.Game import Game
from itertools import chain, combinations
import scipy as sp
from sympy.solvers import solve
from sympy import Symbol


def solve_indifference(A, rows=None, columns=None):
    M = (A[np.array(rows)] - np.roll(A[np.array(rows)], 1, axis=0))[:-1]
    zero_columns = set(range(A.shape[1])) - set(columns)
    if zero_columns != set():
        M = np.append(M, [[int(i == j) for i, col in enumerate(M.T)] for j in zero_columns], axis=0)
    M = np.append(M, np.ones((1, M.shape[1])), axis=0)
    b = np.append(np.zeros(len(M) - 1), [1])
    try:
        prob = np.linalg.solve(M, b)
        if all(prob >= 0):
            print('Prob')
            print(M, b)
            print(prob)
            return prob
        return False
    except np.linalg.LinAlgError:
        return False

def obey_support(strategy, support):
    if strategy is False:
        return False
    if not all((i in support and value >= 0) or (i not in support and value <= 0) for i, value in enumerate(strategy)):
        return False
    return True

def is_ne(payoff1, payoff2, strategy_pair, support_pair):
    u = strategy_pair[1].reshape(strategy_pair[1].size, 1)
    row_payoffs = np.dot(payoff1, u)
    print(u)
    print(row_payoffs)

    v = strategy_pair[0].reshape(strategy_pair[0].size, 1)
    column_payoffs = np.dot(payoff2.T, v)
    print(v)
    print(column_payoffs)

    row_support_payoffs = row_payoffs[np.array(support_pair[0])]
    column_support_payoffs = column_payoffs[np.array(support_pair[1])]
    print(row_support_payoffs)
    print(column_support_payoffs)

    return (row_payoffs.max() == row_support_payoffs.max() and column_payoffs.max() == column_support_payoffs.max())

A = Game()
A2 = np.asarray([[-10, 10, 8, -9],
                 [3, -10, -5, 3],
                 [3, -7, 9, -4],
                 [-4, -1, 4, 1]])
A.matrix = np.asarray(A2)
A.matrix2 = np.asarray(A2*-1)

print(A.matrix)
print(A.matrix2)
print(solve_using_nggw(A.matrix, A.matrix2))
print(reduce_matrix(A.matrix, A.matrix2))

power1 = chain.from_iterable(combinations(range(A.matrix.shape[0]), r) for r in range(A.matrix.shape[0] + 1))
power2 = chain.from_iterable(combinations(range(A.matrix.shape[1]), r) for r in range(A.matrix.shape[1] + 1))

def potential_support_pairs():
    p1_num_strategies, p2_num_strategies = A.matrix.shape
    for support1 in (s for s in power1 if len(s) > 0):
        for support2 in (s for s in power2 if len(s) == len(support1)):
            yield support1, support2
pairs = potential_support_pairs()
def indifference_strategies():
    for pair in pairs:
        s1 = solve_indifference(A.matrix2.T, *(pair[::-1]))
        print(s1)
        s2 = solve_indifference(A.matrix, *pair)
        print(s2)

        if obey_support(s1, pair[0]) and obey_support(s2, pair[1]):
            yield s1, s2, pair[0], pair[1]


#for support1 in (s for s in power1 if len(s) > 0):
#    for support2 in (s for s in power2 if len(s) == len(support1)):
#        pairs.append((support1, support2))
print('Paare:')
print(pairs)
print('potential')
pot_strats = []
for pair in pairs:
    s1 = solve_indifference(A.matrix2.T, *(pair[::-1]))
    print(s1)
    s2 = solve_indifference(A.matrix, *pair)
    print(s2)

    if obey_support(s1, pair[0]) and obey_support(s2, pair[1]):
        pot_strats.append((s1, s2, pair[0], pair[1]))
pot_strats = indifference_strategies()
print('potential')
for pot in pot_strats:
    print(pot)
print('evaluation')
for s1, s2, sup1, sup2 in pot_strats:
    print(s1, s2, sup1, sup2)
    if is_ne(A.matrix, A.matrix2, (s1, s2), (sup1, sup2)):
        print(s1, s2)

def equi():
    return((s1, s2) for s1, s2, sup1, sup2 in indifference_strategies() if is_ne(A.matrix, A.matrix2,(s1, s2), (sup1, sup2)))

equilib = equi()
for eq in equilib:
    print(eq)

# a = np.array([[-10, 10, 8, -9, -1], [3, -10, -5, 3, -1], [3, -7, 9, -4, -1], [-4, -1, 4, 1, -1], [1, 1, 1, 1, 0]])
# b = np.array([0,0,0,0,1])
# print(np.linalg.solve(a,b))
# a = np.array([[-10, 10, 8, -9, -1], [3, -10, -5, 3, -1], [3, -7, 9, -4, -1], [-4, -1, 4, 1, -1], [1, 1, 1, 1, 0]])
# b = np.array([0,0,0,0,1])
# print(np.linalg.lstsq(a,b))
# print(sp.optimize.nnls(a,b))
# print(sp.optimize.lsq_linear(a, b, bounds=([0, 0, 0, 0, -np.inf], [1, 1, 1, 1, np.inf]), tol=0, verbose=1))

x1 = Symbol('x1', nonnegative=True)
x2 = Symbol('x2', nonnegative=True)
x3 = Symbol('x3', nonnegative=True)
x4 = Symbol('x4', nonnegative=True)
w = Symbol('w')
sym = [x1, x2, x3, x4, w]
func1 = [[-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, 3*x1-7*x2+9*x3-4*x4-w, -4*x1-1*x2+4*x3+1*x4-w, x1+x2+x3+x4-1]] # Auszahlung Spieler 1, Strategien 2
func2 = [[10*x1-3*x2-3*x3+4*x4-1*w, -10*x1+10*x2+7*x3+1*x4-w, -8*x1+5*x2-9*x3-4*x4-w, +9*x1-3*x2+4*x3-1*x4-w, x1+x2+x3+x4-1]] # Auszahlung Spieler 2, Strategien 1

init_solve1 = solve(func1[0], dict=True, check=False, force=True)
init_solve2 = solve(func2[0], dict=True, check=False, force=True)

print(init_solve1)
print(init_solve2)
print(init_solve1[0])
negatives = []
for i in range(len(sym)):
    if init_solve1[0][sym[i]] < 0 and sym[i] != w:
        print(sym[i])
        negatives.append(i)
print(negatives)
print(func2)
temp = []
for negative in negatives:
    temp.append(np.delete(func2[0], negative).tolist())
#next_solve2 = np.delete(func2[0], negatives).tolist()
func2.append(deepcopy(temp))
print(func2)
#print(next_solve2)

del negatives[:]
for i in range(len(sym)):
    if init_solve2[0][sym[i]] < 0 and sym[i] != w:
        print(sym[i])
        negatives.append(i)
print(negatives)
del temp[:]
for negative in negatives:
    temp.append(np.delete(func1[0], negative).tolist())
#next_solve1 = np.delete(func1[0], negatives).tolist()
#print(next_solve1)
func1.append(deepcopy(temp))
print(func1)
print(func2)
next_solve1 = []
for funcs in func1[1]:
    next_solve1.append(solve(funcs, dict=True, check=False, force=True))
print(next_solve1)
next_solve2 = []
print(func2[1])
for funcs in func2[1]:
    next_solve2.append(solve(funcs, dict=True, check=False, force=True))
print(next_solve2)

del temp[:]
sols = []
for var in sym:
    for funcs in func2[1]:
        if var != w:
            temp = deepcopy(funcs)
            temp.append(var)
            print(temp)
            temp_sol = solve(temp, dict=True, check=False, force=True)
            print(temp_sol)
            sols.append([temp, temp_sol])

for sol in sols:
    print(sol)
    negatives = []
    for i in range(len(sym)):
        if sol[1][0][sym[i]] < 0 and sym[i] is not w:
            negatives.append(i)

    sol.append(negatives)

print(sols)

print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, 3*x1-7*x2+9*x3-4*x4-w, -4*x1-1*x2+4*x3+1*x4-w, x1+x2+x3+x4-1], dict=True, check=False, force=True))
print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, 3*x1-7*x2+9*x3-4*x4-w, -4*x1-1*x2+4*x3+1*x4-w, x1+x2+x3+x4-1, x1], dict=True, check=False, force=True))

print(solve([3*x1-10*x2-5*x3+3*x4-w, 3*x1-7*x2+9*x3-4*x4-w, -4*x1-1*x2+4*x3+1*x4-w, x1+x2+x3+x4-1, x3], dict=True, check=False, force=True))
print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-7*x2+9*x3-4*x4-w, -4*x1-1*x2+4*x3+1*x4-w, x1+x2+x3+x4-1, x3], dict=True, check=False, force=True))
print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, -4*x1-1*x2+4*x3+1*x4-w, x1+x2+x3+x4-1, x3], dict=True, check=False, force=True))
print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, 3*x1-7*x2+9*x3-4*x4-w, x1+x2+x3+x4-1, x3], dict=True, check=False, force=True))
print()
print()
print(solve([10*x1-3*x2-3*x3+4*x4-1*w, -10*x1+10*x2+7*x3+1*x4-w, -8*x1+5*x2-9*x3-4*x4-w, +9*x1-3*x2+4*x3-1*x4-w, x1+x2+x3+x4-1], dict=True, check=False))
print(solve([10*x1-3*x2-3*x3+4*x4-1*w, -10*x1+10*x2+7*x3+1*x4-w, -8*x1+5*x2-9*x3-4*x4-w, x1+x2+x3+x4-1, x2], dict=True, check=False))
print(solve([10*x1-3*x2-3*x3+4*x4-1*w, -10*x1+10*x2+7*x3+1*x4-w, +9*x1-3*x2+4*x3-1*x4-w, x1+x2+x3+x4-1, x2], dict=True, check=False))
print(solve([10*x1-3*x2-3*x3+4*x4-1*w, -8*x1+5*x2-9*x3-4*x4-w, +9*x1-3*x2+4*x3-1*x4-w, x1+x2+x3+x4-1, x2], dict=True, check=False))
print(solve([-10*x1+10*x2+7*x3+1*x4-w, -8*x1+5*x2-9*x3-4*x4-w, +9*x1-3*x2+4*x3-1*x4-w, x1+x2+x3+x4-1, x2], dict=True, check=False))
#print(solve([10*x1-3*x2-3*x3+4*x4-1*w, -10*x1+10*x2+7*x3+1*x4-w, -8*x1+5*x2-9*x3-4*x4-w, +9*x1-3*x2+4*x3-1*x4-w, x1+x2+x3+x4-1, x4], dict=True, check=False))
#print(solve([10*x1-3*x2-3*x3+4*x4-1*w, -10*x1+10*x2+7*x3+1*x4-w, -8*x1+5*x2-9*x3-4*x4-w, +9*x1-3*x2+4*x3-1*x4-w, x1+x2+x3+x4-1, x3, x4], dict=True, check=False))
#print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, -4*x1-1*x2+4*x3+1*x4-w, x1+x2+x3+x4-1], dict=True, check=False, force=True))
#print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, 3*x1-7*x2+9*x3-4*x4-w, x1+x2+x3+x4-1], dict=True, check=False, force=True))
#print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, -4*x1-1*x2+4*x3+1*x4-w, x1+x2+x3+x4-1, x1], dict=True, check=False, force=True))
#print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, -4*x1-1*x2+4*x3+1*x4-w, x1+x2+x3+x4-1, x1, x3], dict=True, check=False, force=True))
#
# print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, 3*x1-7*x2+9*x3-4*x4-w, x1+x2+x3+x4-1, x1], dict=True, check=False, force=True))
#
# print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, -4*x1-1*x2+4*x3+1*x4-w, x1+x2+x3+x4-1, x2], dict=True, check=False, force=True))
# print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, -4*x1-1*x2+4*x3+1*x4-w, x1+x2+x3+x4-1, x2, x4], dict=True, check=False, force=True))
#
# print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, 3*x1-7*x2+9*x3-4*x4-w, x1+x2+x3+x4-1, x2], dict=True, check=False, force=True))
# print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, 3*x1-7*x2+9*x3-4*x4-w, x1+x2+x3+x4-1, x2, x1], dict=True, check=False, force=True))
#
# print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, -4*x1-1*x2+4*x3+1*x4-w, x1+x2+x3+x4-1, x3], dict=True, check=False, force=True))
#
# print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, 3*x1-7*x2+9*x3-4*x4-w, x1+x2+x3+x4-1, x3], dict=True, check=False, force=True))
#
# print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, -4*x1-1*x2+4*x3+1*x4-w, x1+x2+x3+x4-1, x4], dict=True, check=False, force=True))
#
# print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, 3*x1-7*x2+9*x3-4*x4-w, x1+x2+x3+x4-1, x4], dict=True, check=False, force=True))
# print(solve([-10*x1+10*x2+8*x3-9*x4-1*w, 3*x1-10*x2-5*x3+3*x4-w, 3*x1-7*x2+9*x3-4*x4-w, x1+x2+x3+x4-1, x4, x3], dict=True, check=False, force=True))
#
#
#
# print(solve_using_nggw(A.matrix, A.matrix2))
#
# print(solve([10*x1-3*x2-3*x3+4*x4-1*w, -10*x1+10*x2+7*x3+1*x4-w, +9*x1-3*x2+4*x3-1*x4-w, x1+x2+x3+x4-1], dict=True))
# print(solve([10*x1-3*x2-3*x3+4*x4-1*w, -10*x1+10*x2+7*x3+1*x4-w, +9*x1-3*x2+4*x3-1*x4-w, x1+x2+x3+x4-1, x1], dict=True, check=False))
# print(solve([10*x1-3*x2-3*x3+4*x4-1*w, -10*x1+10*x2+7*x3+1*x4-w, +9*x1-3*x2+4*x3-1*x4-w, x1+x2+x3+x4-1, x1, x2], dict=True, check=False))
# print(solve([10*x1-3*x2-3*x3+4*x4-1*w, -10*x1+10*x2+7*x3+1*x4-w, +9*x1-3*x2+4*x3-1*x4-w, x1+x2+x3+x4-1, x2], dict=True, check=False))
# print(solve([10*x1-3*x2-3*x3+4*x4-1*w, -10*x1+10*x2+7*x3+1*x4-w, +9*x1-3*x2+4*x3-1*x4-w, x1+x2+x3+x4-1, x3], dict=True, check=False))
# print(solve([10*x1-3*x2-3*x3+4*x4-1*w, -10*x1+10*x2+7*x3+1*x4-w, +9*x1-3*x2+4*x3-1*x4-w, x1+x2+x3+x4-1, x3, x4], dict=True, check=False))
# print(solve([10*x1-3*x2-3*x3+4*x4-1*w, -10*x1+10*x2+7*x3+1*x4-w, +9*x1-3*x2+4*x3-1*x4-w, x1+x2+x3+x4-1, x4], dict=True, check=False))
#
# print(solve([-10*x1+10*x2-9*x4-1*w, 3*x1-7*x2-4*x4-w, -4*x1-1*x2+1*x4-w, x1+x2+x4-1], dict=True))

supports = chain.from_iterable(combinations(range(A.matrix.shape[0]), r) for r in range(A.matrix.shape[0] + 1))
supports2 = chain.from_iterable(combinations(range(A.matrix.shape[1]), r) for r in range(A.matrix.shape[1] + 1))

def powerset(n):
    return chain.from_iterable(combinations(range(n), r) for r in range(n+1))

#for supp in supports:
#    print(supp)
#    for strat in supp:
#        print(func1[0][strat])

shape1, shape2 = A.matrix.shape

possible_sols = []

for supp2 in (s for s in powerset(shape2) if len(s) > 0):
    for supp in (s for s in powerset(shape1) if len(s) == len(supp2)):
        print(supp2)
        print(supp)
        temp = []
        strats = [x1+x2+x3+x4-1]
        for strat in supp2:
            strats.append(func2[0][strat])
        for i in range(len(sym)-1):
            if i not in supp:
                strats.append(sym[i])
        solu = solve(strats, dict=True, check=False, force=True)
        solution = True
        temp.append(strats)
        if solu:
            for key in solu[0]:
                if solu[0][key] < 0 and key != w:
                    solution = False
        if solution:
            temp.append(solu)
            temp.append([supp, supp2])
            possible_sols.append(temp)

print(possible_sols)
print(func2[0])
values = []
for sol in possible_sols:
    print(sol)
    print(sol[0])
    print(sol[1])
    print(sol[2])
    temp_funcs = []
    for i in range(shape2):
        if i not in sol[2][1]:
            temp_funcs.append(func2[0][i])
    if temp_funcs:
        temp_funcs.append(x1+x2+x3+x4-1)
        for symb in sym:
            if symb != w and sol[1]:
                temp_funcs.append(symb-sol[1][0][symb])
    other_sol = solve(temp_funcs, dict=True, check=False, force=True)
    optimal = False
    if (sol[1] and other_sol):
        if sol[1][0][w] >= other_sol[0][w]:
            optimal = True
    if optimal:
        print(other_sol[0][w], sol[1][0][w])
        print(temp_funcs)
        values.append(sol[1][0][w])
        print('Optimum gefunden')

print(max(values))