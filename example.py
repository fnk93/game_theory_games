from GameTheory_Game.Game import Game
from GameTheory_Game.Solve import Solve
import numpy as np

example = Game(10, 5, 5)
print(example.get_matrix())
solver = Solve(example)
solver.output()

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