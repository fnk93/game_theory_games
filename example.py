from GameTheory_Game.Game import Game
from GameTheory_Game.Solve import Solve

example = Game(10, 5, 5)
print(example.get_matrix())
solver = Solve(example)
solver.output()