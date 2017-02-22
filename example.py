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
#example.set_matrix(np.array([[0, -1, 2],
#                             [2, 0, -1],
#                             [-1, 2, 0]]))
solver = Solve(example)
solver.output()
x, y, z = sy.symbols("x y z", real=True, positive=True)
w = sy.symbols("w", real=True)
equations = [
    sy.Eq(5*x - 9*y + z, w),
    sy.Eq(-5*x - 5*z, w),
    sy.Eq(-10*x + 8*y - 5*z, w),
    sy.Eq(x + y + z, 1),
]
'''
equations = [
    sy.Eq(0*x - 1*y + 2*z, w),
    sy.Eq(2*x + 0*y - 1*z, w),
    sy.Eq(-1*x + 2*y + 0*z, w),
    sy.Eq(x + y + z, 1),
]
'''
print(equations)
print(sy.solve(equations))


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