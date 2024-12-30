# Задание:
# Измерение расстояния между объектами
# Нарисуйте в осях признаков Х и У точки (записи БД).
# Определите визуально, какие из точек наиболее отдалены от центра класса и являются
# претендентами на образование нового класса или могут являться аномалиями.

import math
import numpy as np
import pandas as pd
from itertools import combinations

def euclidean_norm(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Точки должны иметь одинаковую размерность.")

    squared_diffs = [(a - b) ** 2 for a, b in zip(point1, point2)]
    return math.sqrt(sum(squared_diffs))

data = {
    'Номер точки': [1, 2, 3, 4, 5, 6, 7],
    'X': [2, 3, 1, 4, 1, 5, 4],
    'Y': [4, 2, 1, 3, 6, 3, 2]
}

df = pd.DataFrame(data)
critical_distance = 3

def euclidean_distance(point1, point2):
    return np.sqrt((point1['X'] - point2['X'])**2 +
                   (point1['Y'] - point2['Y'])**2)



results = []

for (i, point1), (j, point2) in combinations(df.iterrows(), 2):
    distance = euclidean_distance(point1, point2)



    results.append((point1['Номер точки'], point2['Номер точки'], distance))

    # if distance >= critical_distance:
    #     results.append((point1['Номер точки'], point2['Номер точки'], distance))

# Выводим результаты
# for point1, point2, distance in results:
#     print(f'Расстояние между точками {point1} и {point2}: {distance:.2f}')


print(results)