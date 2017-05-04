import numpy as np

payoff_points = np.asarray([[80, 80],
                            [0, 120],
                            [120, 0],
                            [10, 10]])

sorted_payoff_points = payoff_points[0]

for i in range(0, len(payoff_points)-1):
    min_distance = list()
    print(i)
    for j in range(i+1, len(payoff_points)):
        print(j)
        min_distance.append(abs(payoff_points[i][0]-payoff_points[j][0]) + abs(payoff_points[i][1]-payoff_points[j][1]))
    print('minima: ', np.argmin(min_distance)+i+1)

    payoff_points[i+1], payoff_points[np.argmin(min_distance)+i+1] = payoff_points[np.argmin(min_distance)+i+1], payoff_points[i+1].copy()

    print(payoff_points)

    min_distance.clear()