# Задание 2:
# Задаться примером. Методом «Косинуса» провести анализ классов, вывести дерево классов, правила,
# провести прогноз на модели, нарисовать геометрическую иллюстрацию, выбрать датчики, привести регрессионную модель.
# Например, найдите классы автомобилей.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 0-КР, 1-ВН, 2-П, 3-Х, 4-С, 5-МВ, 6-РД, 7-КБ, 8-КП, 9-Л, 10-В, 11-МК, 12-СК, 13-МА, 14-КМ

data = {
    'Масса (т.)': [2.3, 3.5, 3, 1.6, 2, 1.8, 1.5, 1.5, 1.6, 2.5, 2, 0.7, 1.7, 2.5, 4],
    'Длинна (м.)': [4.7, 5, 5.5, 4, 4.6, 5.2, 4, 4, 3.7, 7, 5.2, 3.3, 4.3, 7.5, 8],
    'Объем двигателя (л.)': [2.5, 4, 3.5, 1.6, 2, 2.5, 1.8, 1.6, 1.5, 7, 4, 1, 6, 4, 6],
    'Расход топлива (л/100км)': [12, 15, 13, 9, 10, 12, 11, 10, 8, 30, 16, 5, 16, 20, 25],
    'Количество мест': [5, 7, 4, 5, 5, 8, 2, 2, 2, 12, 3, 2, 2, 16, 10],
    'Цена (млн. руб.)': [7, 12, 9, 3, 5, 7, 10, 6, 3, 15, 10, 2.5, 20, 12, 10]
}

df = pd.DataFrame(data)

data_names_abbr = ["КР", "ВН", "П", "Х", "С", "МВ", "РД", "КБ", "КП", "Л", "В", "МК", "СК", "МА", "КМ"]
data_names = ["Кроссовер", "Внедорожник", "Пикап", "Хэтчбэк", "Седан", "Минивен", "Роадстер", "Кабриолет",
              "Купе", "Лимузин", "Вен", "Микро", "Спорткар", "Микроавтобус", "Кэмпер"]


# Метод конвертации DataFrame в Список
def convert_df_to_list(dataframe):
    data_list = list()
    transposed_df = dataframe.T

    for i in range(len(dataframe)):
        data_list.append(transposed_df[i].values.tolist())

    return data_list


# Метод вычисления похожести методом косинуса двух объектов
def cos_similarity(vec_a, vec_b):
    # Преобразуем векторы в numpy массивы
    a = np.array(vec_a)
    b = np.array(vec_b)

    # Вычисляем скалярное произведение
    dot_product = np.dot(a, b)

    # Вычисляем нормы векторов
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Вычисляем косинусное сходство
    if norm_a == 0 or norm_b == 0:
        return 0.0  # Избегаем деления на ноль

    return dot_product / (norm_a * norm_b)


# Метод вычисления похожести для набора данных и формирование результата вычисления в файле xlsx
def calculate_data_similarity(dataframe):
    similarity_df = dict()
    converted_df = convert_df_to_list(dataframe)

    for i, target_list_value in enumerate(converted_df):
        result_list = list()
        for target_list_j in converted_df:
            similarity = cos_similarity(target_list_value, target_list_j)
            result_list.append(similarity)

        similarity_df[data_names[i]] = result_list

    return pd.DataFrame(similarity_df)


# Для удобства просмотра таблицы в консоле
pd.options.display.float_format = '{:,.3f}'.format
pd.options.display.max_columns = None

# Вычисляем похожесть
result_similarity = calculate_data_similarity(df)
result_similarity.to_excel('result_similarity.xlsx')
# print(result_similarity)


# Стандартизируем данные
scaler = StandardScaler()
scaled_similarity_data = scaler.fit_transform(result_similarity)

for i in range(2, 5):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(scaled_similarity_data)
    df['k=%d' % i] = kmeans.labels_

df.to_excel('kmeans_df.xlsx')

# Визуализация кластеров k-means
plt.figure(figsize=(10, 6))

sns.scatterplot(data=df, x='Масса (т.)', y='Расход топлива (л/100км)', hue='k=3', palette='viridis', s=100)
plt.title('k-means кластеризация автомобилей по паре свойств Масса - Расход топлива')
plt.xlabel('Масса (т.)')
plt.ylabel('Расход топлива (л/100км)')
plt.legend()
plt.show()

sns.scatterplot(data=df, x='Объем двигателя (л.)', y='Цена (млн. руб.)', hue='k=4', palette='viridis', s=100)
plt.title('k-means кластеризация автомобилей по паре свойств Объем двигателя - Цена')
plt.xlabel('Объем двигателя (л.)')
plt.ylabel('Цена (млн. руб.)')
plt.legend()
plt.show()

sns.scatterplot(data=df, x='Объем двигателя (л.)', y='Расход топлива (л/100км)', hue='k=4', palette='viridis', s=100)
plt.title('k-means кластеризация автомобилей по паре свойств Объем двигателя - Расход топлива')
plt.xlabel('Объем двигателя (л.)')
plt.ylabel('Расход топлива (л/100км)')
plt.legend()
plt.show()

sns.scatterplot(data=df, x='Длинна (м.)', y='Количество мест', hue='k=3', palette='viridis', s=100)
plt.title('k-means кластеризация автомобилей по паре свойств Длинна - Количество мест')
plt.xlabel('Длинна (м.)')
plt.ylabel('Количество мест')
plt.legend()
plt.show()

# Построение дерева классов
linked = linkage(scaler.fit_transform(df), method='ward')
dendrogram(linked, orientation='top', labels=np.array(data_names_abbr), count_sort=True,
           distance_sort='descending', show_leaf_counts=True)
plt.title('Дерево классов автомобилей')
plt.xlabel('Классы автомобилей')
plt.ylabel('К')
plt.show()

# Прогнозирование

# Определим признаки и целевую переменную
x = df[['Масса (т.)', 'Объем двигателя (л.)', 'Количество мест']]
y = df['Расход топлива (л/100км)']

# Разделим данные на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Создадим и обучим модель линейной регрессии
model = LinearRegression()
model.fit(x_train, y_train)

# Сделаем прогнозы на тестовой выборке
y_pred = model.predict(x_test)

# Оценим качество модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Визуализируем результаты
plt.scatter(y_test, y_pred)
plt.xlabel('Фактический расход топлива')
plt.ylabel('Предсказанный расход топлива')
plt.title('Фактический vs Предсказанный расход топлива')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y_test, y_pred, c='b', marker='o')
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
ax.set_xlabel('Масса (т.)')
ax.set_ylabel('Объем двигателя')
ax.set_zlabel('Количество мест')
ax.set_title('Зависимость расхода топлива от массы автомобиля, объема двигателя и количества мест')
plt.show()
