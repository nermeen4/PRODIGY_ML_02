#import necessary libraries 
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


#load our data!!
MC_Data=pd.read_csv('E:\\interships\\prodigy intern\\Mall_Customers.csv')

#show first rows of data 
print(MC_Data.head())

#data preproccesing
x = MC_Data[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#create model of k-mean
k=6
k_mn=KMeans(n_clusters=k, random_state=42)
k_mn.fit(x_scaled)

MC_Data['Cluster'] = k_mn.labels_

#evaluate model
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=MC_Data, palette='viridis', s=100)
plt.scatter(scaler.inverse_transform(k_mn.cluster_centers_)[:, 0], scaler.inverse_transform(k_mn.cluster_centers_)[:, 1], s=300, c='red', label='Centroids', marker='*')
plt.title("Customer Segmentation using K-means Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

#optional part for optimal k 
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_scaled)
    inertia.append(k_mn.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K, inertia, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()