from GameTheory_Game.ZerosumGame import ZerosumGame
from GameTheory_Game.Solving_Methods import *
import numpy as np

A = ZerosumGame
A.matrix = np.asarray([[3,0,2],
                       [4,5,1],
                       [2,2,-1],])
A.matrix2 = A.matrix * -1

sol = get_calculations_latex(A.matrix, A.matrix2, mode=1)[3]
print(sol['first_step'])

for step in sol['simplex_steps']:
    print(step[0])
    print(step[1])
    print(step[2])
    print(step[3])
    print(step[4])

#res = use_simplex(A.matrix, A.matrix2)

#print(res)
