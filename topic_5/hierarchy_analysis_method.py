import numpy as np


# Функция для вычисления среднего геометрического по строкам
def calculation_geometric_means(data):
    geometric_means = np.prod(data, axis=1) ** (1 / data.shape[1])
    return geometric_means


# Функция для вычисления суммы средних геометрических
def calculation_sum_geometric_means(geometric_means):
    sum_geometric_means = np.sum(geometric_means)
    return sum_geometric_means


# Функция для вычисления нормализованного коэффицинта
def calculation_normalized_coefficient(sum_geometric_means, geometric_means):
    normalized_coefficients = list()
    for i, geometric_mean in enumerate(geometric_means, start=1):
        normalized_coefficient = geometric_mean / sum_geometric_means
        normalized_coefficients.append(normalized_coefficient)
    return normalized_coefficients


# Функция для вычисления веса атрибутов
def calculate_attribute_weight(data, coefficients):
    attribute_weight_sum_list = list()
    for i, row in enumerate(data, start=1):
        attribute_weight = float()
        for j, cell in enumerate(row):
            attribute_weight = attribute_weight + (cell * coefficients[j])
        attribute_weight_sum_list = np.append(attribute_weight_sum_list, attribute_weight)
    return attribute_weight_sum_list


# Функция для вычисления приоритетов варианта на основе весовых коэффициентов
def calculate_feature_priorities_weighting_coefficients(high_lvl_data, middle_lvl_datas):
    geometric_means_high_lvl = calculation_geometric_means(high_lvl_data)
    sum_geometric_means_high_lvl = calculation_sum_geometric_means(geometric_means_high_lvl)
    normalized_coefficients_high_lvl = calculation_normalized_coefficient(sum_geometric_means_high_lvl, geometric_means_high_lvl)

    attribute_weight_high_lvl = calculate_attribute_weight(high_lvl_data, normalized_coefficients_high_lvl)
    sum_geometric_means_high_lvl = calculation_sum_geometric_means(attribute_weight_high_lvl)
    normalized_coefficients_high_lvl = calculation_normalized_coefficient(sum_geometric_means_high_lvl, attribute_weight_high_lvl)

    result = np.empty((0, 0))
    for i, data in enumerate(middle_lvl_datas):
        geometric_means_middle_lvl = calculation_geometric_means(data)
        sum_geometric_means_middle_lvl = calculation_sum_geometric_means(geometric_means_middle_lvl)
        normalized_coefficients_middle_lvl = calculation_normalized_coefficient(sum_geometric_means_middle_lvl, geometric_means_middle_lvl)

        attribute_weight_middle_lvl = calculate_attribute_weight(data, normalized_coefficients_middle_lvl)
        sum_geometric_means_middle_lvl = calculation_sum_geometric_means(attribute_weight_middle_lvl)
        normalized_coefficients_middle_lvl = calculation_normalized_coefficient(sum_geometric_means_middle_lvl, attribute_weight_middle_lvl)

        final_normalized_coefficients_middle_lvl = np.dot(np.array(normalized_coefficients_middle_lvl).reshape(-1, 1), normalized_coefficients_high_lvl[i])
        result = np.append(result, final_normalized_coefficients_middle_lvl)
    return result.reshape(-1, 1)


# Функция для вычисления весовых коэффициентов атрибутов
def calculate_attribute_weight_coefficients(data):
    result = np.empty((len(data[0]),len(data)))
    for i, target_data in enumerate(data):
        geometric_means = calculation_geometric_means(target_data)
        sum_geometric_means = calculation_sum_geometric_means(geometric_means)
        normalized_coefficients = calculation_normalized_coefficient(sum_geometric_means, geometric_means)
        for p, val in enumerate(normalized_coefficients):
            result[p][i] = val
    return result


print('__________Расчет_выгоды__________')
# Данные оценок верхнего уровня выгоды A
dataA = np.array([
    [1, 3, 6],
    [1 / 3, 1, 2],
    [1 / 6, 1 / 2, 1]
])

# Данные оценок среднего уровня выгоды В
dataB1 = np.array([
    [1, 1 / 3, 1 / 7, 1 / 5, 1 / 6],
    [3, 1, 1 / 4, 1 / 2, 1 / 2],
    [7, 4, 1, 7, 5],
    [5, 2, 1 / 7, 1, 1 / 5],
    [6, 2, 1 / 5, 5, 1]
])

dataB2 = np.array([
    [1, 6, 9],
    [1 / 6, 1, 4],
    [1 / 9, 1 / 4, 1]
])

dataB3 = np.array([
    [1, 1 / 4, 6],
    [4, 1, 8],
    [1 / 6, 1 / 8, 1]
])

# Данные оценок среднего уровня выгоды С
dataC1 = np.array([
    [1, 2, 7],
    [1 / 2, 1, 6],
    [1 / 7, 1 / 6, 1]
])

dataC2 = np.array([
    [1, 1 / 2, 8],
    [2, 1, 9],
    [1 / 8, 1 / 9, 1]
])

dataC3 = np.array([
    [1, 4, 8],
    [1 / 4, 1, 6],
    [1 / 8, 1 / 6, 1]
])

dataC4 = np.array([
    [1, 1, 6],
    [1, 1, 6],
    [1 / 6, 1 / 6, 1]
])

dataC5 = np.array([
    [1, 1 / 4, 9],
    [4, 1, 9],
    [1 / 9, 1 / 9, 1]
])

dataC6 = np.array([
    [1, 4, 7],
    [1 / 4, 1, 6],
    [1 / 7, 1 / 6, 1]
])

dataC7 = np.array([
    [1, 1, 5],
    [1, 1, 5],
    [1 / 5, 1 / 5, 1]
])

dataC8 = np.array([
    [1, 5, 3],
    [1 / 5, 1, 1 / 3],
    [1 / 3, 3, 1]
])

dataC9 = np.array([
    [1, 5, 8],
    [1 / 5, 1, 5],
    [1 / 8, 1 / 5, 1]
])

dataC10 = np.array([
    [1, 3, 7],
    [1 / 3, 1, 6],
    [1 / 7, 1 / 6, 1]
])

dataC11 = np.array([
    [1, 3, 7],
    [1 / 3, 1, 6],
    [1 / 7, 1 / 6, 1]
])


dataB_arrays = np.array([dataB1, dataB2, dataB3], dtype=object)
dataC_arrays = np.array([dataC1, dataC2, dataC3, dataC4, dataC5, dataC6, dataC7, dataC8, dataC9,
                         dataC10, dataC11], dtype=object)

feature_priorities_weighting_coefficients = calculate_feature_priorities_weighting_coefficients(dataA, dataB_arrays)

attribute_weight_coefficients = calculate_attribute_weight_coefficients(dataC_arrays)
for i, coefficients in enumerate(attribute_weight_coefficients, start=1):
    print(f'Весовые коэффициенты характеристики С для М{i}: {coefficients}')

A = np.dot(attribute_weight_coefficients, feature_priorities_weighting_coefficients)
for i, weight in enumerate(A, start=1):
    print(f'Вес выгоды варианта М{i}: {weight}')


print('__________Расчет_издержек__________')
# Данные оценок верхнего уровня издержек D
dataD = np.array([
    [1, 5, 7],
    [1 / 5, 1, 2],
    [1 / 7, 1 / 2, 1]
])

# Данные оценок среднего уровня издержек E
dataE1 = np.array([
    [1, 7, 9],
    [1/7, 1, 5],
    [1/9, 1/5, 1]
])

dataE2 = np.array([
    [1, 1/3, 1/5],
    [3, 1, 1/5],
    [5, 5, 1]
])

dataE3 = np.array([
    [1, 3, 4],
    [1/3, 1, 1/3],
    [1 / 4, 3, 1]
])

# Данные оценок среднего уровня издержек K
dataK1 = np.array([
    [1, 1/3, 8],
    [3, 1, 9],
    [1 / 8, 1 / 9, 1]
])

dataK2 = np.array([
    [1, 1 /3, 8],
    [3, 1, 9],
    [1 / 8, 1 / 9, 1]
])

dataK3 = np.array([
    [1, 1, 9],
    [3, 1, 9],
    [1 / 9, 1 / 9, 1]
])

dataK4 = np.array([
    [1, 4, 9],
    [1/4, 1, 8],
    [1 / 9, 1 / 8, 1]
])

dataK5 = np.array([
    [1, 1, 9],
    [1, 1, 9],
    [1 / 9, 1 / 9, 1]
])

dataK6 = np.array([
    [1, 1, 9],
    [1, 1, 9],
    [1 / 9, 1 / 9, 1]
])

dataK7 = np.array([
    [1, 3, 8],
    [1/3, 1, 6],
    [1 / 8, 1 / 6, 1]
])

dataK8 = np.array([
    [1, 3, 7],
    [1 / 3, 1, 5],
    [1 / 7, 1/5, 1]
])

dataK9 = np.array([
    [1, 1/6, 7],
    [6, 1, 8],
    [1 / 7, 1 / 8, 1]
])

dataE_arrays = np.array([dataE1, dataE2, dataE3], dtype=object)
dataK_arrays = np.array([dataK1, dataK2, dataK3, dataK4, dataK5, dataK6, dataK7, dataK8, dataK9], dtype=object)

feature_priorities_weighting_coefficients = calculate_feature_priorities_weighting_coefficients(dataD, dataE_arrays)


attribute_weight_coefficients = calculate_attribute_weight_coefficients(dataK_arrays)
for i, coefficients in enumerate(attribute_weight_coefficients, start=1):
    print(f'Весовые коэффициенты характеристики K для М{i}: {coefficients}')

D = np.dot(attribute_weight_coefficients, feature_priorities_weighting_coefficients)
for i, weight in enumerate(D, start=1):
    print(f'Вес издержек варианта М{i}: {weight}')


print('__________Расчет_чистой_выгоды__________')
for i, weight in enumerate(A, start=1):
    print(f'Вес чистой выгоды варианта М{i}: {weight-D[i-1]}')
