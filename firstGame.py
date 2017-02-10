import numpy as np
from operator import add, neg

game1 = [[1, 0],
         [2, 1],
         [0, -1]]
game1_numpy = np.array(game1)
rows = game1_numpy.shape[0]
columns = game1_numpy.shape[1]
print(game1_numpy)

for i in range(rows):
    row = ''
    for j in range(columns):
        row = row + str(game1_numpy[i][j]) + ' '
    print(row)

best_strat_a = 0

strat_array_1 = np.zeros(columns)
strat_array_2 = np.zeros(rows)
print(strat_array_1)
print(strat_array_2)

for i in range(columns):
    start_value_1 = -50
    for j in range(rows):
        if game1_numpy[j][i] > start_value_1:
            start_value_1 = game1_numpy[j][i]
            strat_array_1[i] = j

strat_start_1 = strat_array_1[0]

final_strat_1 = 0
for strat_1 in range(strat_array_1.shape[0]):
    if strat_1 == strat_start_1:
        print('Strategie ' + str(strat_1+1) + ' wird von Spieler 1 gewählt.')
        final_strat_1 = strat_1

for i in range(rows):
    start_value_2 = +50
    for j in range(columns):
        if game1_numpy[i][j] < start_value_2:
            start_value_2 = game1_numpy[i][j]
            strat_array_2[i] = j

#print(strat_array_2)
strat_start_2 = strat_array_2[0]
final_strat_2 = 0
for strat_2 in range(strat_array_2.shape[0]):
    if strat_2 == strat_start_2:
        print('Strategie ' + str(strat_2+1) + ' wird von Spieler 2 gewählt.')
        final_strat_2 = strat_2

print('Auszahlung für Spieler 1: ' + str(game1_numpy[final_strat_1][final_strat_2]))
print('Auszahlung für Spieler 2: ' + str(game1_numpy[final_strat_1][final_strat_2]*(-1)))