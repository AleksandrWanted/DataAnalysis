# Задание 1:
# Задаться примером для метода k-means (не менее 3D, не менее 15 экземпляров).
# Например, задаться количеством признаков, достаточным для отличия кошек от собак и различения их по видам.
# Провести кластеризацию множества экземпляров для случаев: k=2, k=3, k=4. Нарисовать дерево классов.
# Привести графическую иллюстрацию.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# 0-кошка, 1-собака, 2-кошка, 3-собака, 4-кошка, 5-собака, 6-кошка, 7-собака,
# 8-кошка, 9-кошка, 10-кошка, 11-собака, 12-кошка, 13-собака, 14-кошка

data = {
    'Вес, в кг.': [4, 20, 3.5, 15, 5, 25, 4.5, 18, 3, 10, 2.5, 22, 3.8, 30, 4.2],
    'Рост, в см.': [20, 50, 18, 35, 25, 52, 27, 38, 15, 28, 14, 50, 20, 60, 18],
    'Длинна шерсти, в мм.': [10, 40, 12, 8, 15, 70, 1, 25, 13, 20, 50, 80, 30, 100, 50],
    'Длина хвоста, в см.': [25, 30, 20, 4, 18, 27, 35, 10, 17, 18, 7, 35, 20, 15, 16]
}


data_names_abbr = ["кошка", "собака", "кошка", "собака", "кошка", "собака", "кошка",
                   "собака", "кошка", "кошка", "кошка", "собака", "кошка", "собака", "кошка"]

# Преобразуем в DataFrame
df = pd.DataFrame(data)


# Функция для кластеризации k-means
def kmeans_clustering(df, k):
    kmeans = KMeans(n_clusters=k)
    df['k=%d' % k] = kmeans.fit_predict(
        df[['Вес, в кг.', 'Рост, в см.', 'Длинна шерсти, в мм.', 'Длина хвоста, в см.']])
    df.to_excel('./kmeans_df.xlsx')


for k in [2, 3, 4]:
    kmeans_clustering(df, k)

# Стандартизируем данные
scaler = StandardScaler()
scaled_similarity_data = scaler.fit_transform(df)

# Построение дерева классов
linked = linkage(scaler.fit_transform(df), method='ward')
dendrogram(linked, orientation='top', labels=np.array(data_names_abbr), count_sort=True,
           distance_sort='descending', show_leaf_counts=True)
plt.title('Кошки и собаки')
plt.xlabel('Тип животного')
plt.ylabel('К')
plt.show()