import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("sizekorea2010.csv")
x = dataset.iloc[:, [2, 4, 7, 15, 17, 18, 19, 20, 22]].values

x_train, x_test = train_test_split(x, test_size=0.2, random_state=0)
x_origin = x_train
# x_train = normalize(x_train, norm='max', axis=0)
'''x_train=sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
'''
from sklearn.cluster import KMeans

'''
#within cluster sum of squared
wcss=[]
for i in range(1, 11):
   kmeans = KMeans(n_clusters =i, init ='k-means++', random_state=42)
   kmeans.fit(x_train)
   wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Chair distribute')
plt.xlabel('Num of Cluster')
plt.ylabel('WCSS')
plt.show()
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x_train, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('case')
plt.ylabel('Euclidean distances')
plt.show()
'''

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(x_train)
plt.scatter(x_train[y_kmeans == 0, 5], x_train[y_kmeans == 0, 3], s=100, c='red', label='Cluster 1')
plt.scatter(x_train[y_kmeans == 1, 5], x_train[y_kmeans == 1, 3], s=100, c='blue', label='Cluster 2')
plt.scatter(x_train[y_kmeans == 2, 5], x_train[y_kmeans == 2, 3], s=100, c='green', label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 5], kmeans.cluster_centers_[:, 3], s=300, c='yellow', label='Centroids')
plt.title('Clusters of Chair')
plt.xlabel('height')
plt.ylabel('hip depth')
plt.legend()
plt.show()

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(x_train)

# Visualising the clusters
plt.scatter(x_train[y_hc == 0, 0], x_train[y_hc == 0, 4], s=100, c='red', label='Cluster 1')
plt.scatter(x_train[y_hc == 1, 0], x_train[y_hc == 1, 4], s=100, c='blue', label='Cluster 2')
plt.scatter(x_train[y_hc == 2, 0], x_train[y_hc == 2, 4], s=100, c='green', label='Cluster 3')
'''
plt.scatter(x_train[y_hc == 3, 0], x_train[y_hc == 3, 4], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x_train[y_hc == 4, 0], x_train[y_hc == 4, 4], s = 100, c = 'magenta', label = 'Cluster 5')
'''
plt.title('Clusters by AgglomerativeClustering')
plt.xlabel('1')
plt.ylabel('2')
plt.legend()
plt.show()


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


print(np.max(x_train[y_kmeans == 0, 1]))
mat = np.zeros((6, 9))
for i in range(3):
    for j in range(9):
        # print(i,j)
        a = reject_outliers(x_train[y_kmeans == i, j])
        mat[2 * i, j] = np.max(a)
        mat[2 * i + 1, j] = np.min(a)

print(mat)
np.savetxt("manmin_filtered.csv", mat, delimiter=",")