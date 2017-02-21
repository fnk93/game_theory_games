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


example = Game()
print(example.get_matrix())
#example.set_matrix(np.array([[0, -1, 2],
#                             [2, 0, -1],
#                             [-1, 2, 0]]))
solver = Solve(example)
solver.output()

#Eqns = [x + y = 0, x >= 0, y >= 0]
x= symbols('x')
y = symbols('y')
eq1 = Relational(x+y, 1, '==')
eq1 = ((Poly(x+y), Poly(1, x, y)), '==')
eq2 = Relational(x, 0, '>=')
eq2 = ((Poly(x), Poly(0, x)), '>=')
eq3 = Relational(y, 0, '>=')
eq3 = ((Poly(y), Poly(0, y)), '>=')
eq1 = ((Poly(x+y), Poly(1, x,y)), '>=')
z = symbols('z')
w2 = symbols('w2')
print([Eq(5*x+-9*y + z, w2), Eq(-5*x - 5*z, w2).as_expr(), Eq(-10*x +8*y-5*z, w2).as_expr(), Eq(x+y+z,1).as_expr(), GreaterThan(x, 0), Ge(y, 0), Ge(z,0)], [x, y, z, w2])
print(linsolve([Eq(5*x+-9*y + z, w2), Eq(-5*x - 5*z, w2), Eq(-10*x +8*y-5*z, w2), Eq(x+y+z,1), Ge(x, 0), Ge(y, 0), Ge(z,0)], [x, y, z, w2]))
print(linsolve([Eq(x+y,1), Eq(2*x+y, 2), Ge(x, 0), Ge(y, 0)], [x,y]))
print(linsolve([Eq(5*x+-9*y + z, w2), Eq(-5*x - 5*z, w2), Eq(-10*x +8*y-5*z, w2), Eq(x+y+z,1), GreaterThan(x, 0), GreaterThan(y, 0), GreaterThan(z,0)], [x, y, z, w2]))
print(np.linalg.solve([[5, -9, 1, -1], [-5, 0, -5, -1], [-10, 8, -5, -1], [1, 1, 1, 0]], [0, 0, 0, 1]))
x, y, z, w = sy.symbols("x y z w")
equations = [
    sy.Eq(5*x - 9*y + z, w),
    sy.Eq(-5*x - 5*z, w),
    sy.Eq(-10*x + 8*y - 5*z, w),
    sy.Eq(x + y + z, 1),
]

print(reduce_inequalities([
        sy.Ge(1*x, 0),
        sy.Ge(1*y, 0),
        sy.Ge(1*z, 0),
        ]))
print(sy.solve(equations))
equations2 = [
    sy.Eq(5*x - 9*y, w),
    sy.Eq(-5*x, w),
    sy.Eq(-10*x + 8*y, w),
    sy.Eq(x + y, 1),
]
print(sy.solve(equations2))

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