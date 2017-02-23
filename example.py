from GameTheory_Game.Game import Game
from GameTheory_Game.Solve import Solve
import numpy as np
from sympy import Matrix, S, linsolve, symbols, Poly
from sympy.solvers.inequalities import reduce_inequalities
from sympy import Ge, Eq
from sympy.core.relational import Relational
from sympy import *
from scipy import optimize
import sympy as sy
from sympy import Abs


example = Game()
print(example.get_matrix())
#example.set_matrix(np.array([[0, 4, -10, 9],
#                             [-2, 7, 8, -9],
#                             [3, -7, -9, 0],
#                             [9, 8, 1, -2]]))
#example.set_matrix(np.array([[10, 5],
#                             [-7, -3],
#                             [3, -7]]))
print(example.get_matrix())
solver = Solve(example)
solver.output()

variablen = var('q:3, w')
for count in range(len(variablen)):
    print(variablen[count])

'''
w = sy.symbols('w', real=True)
w2 = symbols('w2', real=True)
v, x, y, z = sy.symbols("v x y z", real=True, positive=True)
equations = [
    sy.Eq(-10*x + 5*y - 5*z, w),
    sy.Eq(9*x - 5*y + 6*z, w),
    sy.Eq(-6*x - 2*y + 7*z, w),
    sy.Eq(x + y + z, 1),
]
print('Reduziertes Spiel und Strategien Spieler 1')
print(sy.solve(equations))
print(23/40 + 43/320 + 93/320)
equations2 = [
    sy.Eq(10*x - 9*y + 6*z, w),
    sy.Eq(-5*x + 5*y + 2*z, w),
    sy.Eq(5*x - 6*y - 7*z, w),
    sy.Eq(x + y + z, 1),
]
print('Reduziertes Spiel und Strategien Spieler 2')
print(sy.solve(equations2))
equations3 = [
    sy.Eq(-10*x + 5*y - 5*z + 5*v, w),
    sy.Eq(9*x - 5*y + 6*z - 5*v, w),
    sy.Eq(-6*x - 2*y + 7*z + 6*v, w),
    sy.Eq(x + y + z + v, 1),
]
print('Nicht Reduziertes Spiel und Strategien Spieler 1')
print(sy.solve(equations3))
equations4 = [
    sy.Eq(10*x - 9*y + 6*z, w),
    sy.Eq(-5*x + 5*y + 2*z, w),
    sy.Eq(5*x - 6*y - 7*z, w),
    sy.Eq(-5*x + 5*y - 6*z, w),
    sy.Eq(x + y + z, 1),
]
print('Nicht Reduziertes Spiel und Strategien Spieler 2')
print(sy.solve(equations4))
'''
'''
game1 = [[-10, -3, 10, 2],
         [2, 0, 1, -2]]
game1_numpy = np.array(game1)
example2 = Game(10, 5, 5)
example2.set_matrix(game1_numpy)
print(example2.get_matrix())
solver2 = Solve(example2)
solver2.output()
'''