# Задание 1:
# Задаться примером для метода k-means (не менее 3D, не менее 15 экземпляров).
# Провести кластеризацию множества экземпляров для случаев: k=2, k=3, k=4.
# Нарисовать дерево классов.
# Привести графическую иллюстрацию.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# 0-КР, 1-ВН, 2-П, 3-Х, 4-С, 5-МВ, 6-РД, 7-КБ, 8-КП, 9-Л, 10-В, 11-МК, 12-СК, 13-МА, 14-КМ

data = {
    'Объем двигателя (л.)': [2.5, 4, 3.5, 1.6, 2, 2.5, 1.8, 1.6, 1.5, 7, 4, 1, 6, 4, 6],
    'Мощность (л.с.)': [230, 270, 250, 150, 180, 160, 350, 300, 250, 350, 140, 100, 500, 120, 180],
    'Расход топлива (л/100км)': [12, 15, 13, 9, 10, 12, 11, 10, 8, 30, 16, 5, 16, 20, 25],
    'Вес (т.)': [2.3, 3.5, 3, 1.6, 2, 1.8, 1.5, 1.5, 1.6, 2.5, 2, 0.7, 1.7, 2.5, 4]
}

df = pd.DataFrame(data)

data_names_abbr = ["КР", "ВН", "П", "Х", "С", "МВ", "РД", "КБ", "КП", "Л", "В", "МК", "СК", "МА", "КМ"]


# Функция для кластеризации k-means
def kmeans_clustering(df, k):
    kmeans = KMeans(n_clusters=k)
    df['k=%d' % k] = kmeans.fit_predict(
        df[['Объем двигателя (л.)', 'Мощность (л.с.)', 'Расход топлива (л/100км)', 'Вес (т.)']])
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
plt.title('Дерево классов автомобилей')
plt.xlabel('Классы автомобилей')
plt.ylabel('К')
plt.show()
