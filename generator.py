import numpy as np
from random import randrange
from scipy import optimize

lines = randrange(2, 5)
print('Zeilen: ' + str(lines))
columns = randrange(2, 5)
print('Spalten: ' + str(columns))

new_game = np.zeros((lines, columns))
print(new_game)

# Annahme: Symmetrische Beschränkung der Auszahlung
max_int = 10
min_int = -10
#max_restrict = (max_int * 2) + 1

for count_lin in range(lines):
    for count_col in range(columns):
        x = randrange(min_int, max_int+1)
#        new_game[count_lin][count_col] = x - max_int
        new_game[count_lin][count_col] = x

print(new_game)

#test_game = [[0, -1, 2],
#             [2, 0, -1],
#             [-1, 2, 0]]

#new_game = np.asarray(test_game)
#lines = new_game.shape[0]
#columns = new_game.shape[1]

most_negative = np.amin(new_game)
print(most_negative)

game_for_simplex = []

if most_negative < 1:
    game_for_simplex = new_game - (most_negative - 1)
else:
    game_for_simplex = new_game

print(game_for_simplex)

c = []
for count_lin in range(lines):
    c.append(1)

A = []
for count_col in range(columns):
    temp = []
    for count_lin in range(lines):
        temp.append(game_for_simplex[count_lin][count_col] * -1)
    A.append(temp)

b = []
for count_col in range(columns):
    b.append(-1)

game_bounds = []
for count_lin in range(lines):
    game_bounds.append((0, None))

print(c)
print(A)
print(b)
print(game_bounds)

result = optimize.linprog(c, A, b, bounds=game_bounds)
print(result)

print('Spielwert: \n' + str(1/result.fun + (most_negative - 1)))

print(len(result.x))

for y in range(len(result.x)):
    print('Spieler 1: Wahrscheinlichkeit Strategie ' + str(y) + ': ' + str(result.x[y] * (1/result.fun)))

c2 = []
for count_col in range(columns):
    c2.append(-1)

A2 = []
for count_lin in range(lines):
    temp = []
    for count_col in range(columns):
        temp.append(game_for_simplex[count_lin][count_col])
    A2.append(temp)

b2 = []
for count_lin in range(lines):
    b2.append(1)

game_bounds2 = []
for count_col in range(columns):
    game_bounds2.append((0, None))

result2 = optimize.linprog(c2, A2, b2, bounds=game_bounds2)
print(c2)
print(A2)
print(b2)
print(game_bounds2)
print(result2)

print('Spielwert: \n' + str(1/result2.fun - (most_negative - 1)))

for y in range(len(result2.x)):
    print('Spieler 2: Wahrscheinlichkeit Strategie ' + str(y) + ': ' + str(abs(result2.x[y] * (1/result2.fun))))