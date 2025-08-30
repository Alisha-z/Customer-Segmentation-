import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("Mall_Customers.csv")
data.head(5)

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ssd = []
k_values = range(1, 15)
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_scaled)
    ssd.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, ssd, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances (SSD)')
plt.title('Elbow Method for Optimal k')
plt.grid()
plt.show()

kmeans = KMeans(n_clusters=5)
data['cluster'] = kmeans.fit_predict(X_scaled)

average_spending = data.groupby('cluster')['Spending Score (1-100)'].mean().reset_index()
average_spending.columns = ['cluster', 'average_spending']

print(average_spending)

sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='cluster')
plt.title('Customer Segmentation')
plt.show()

dbscan = DBSCAN(eps=0.5, min_samples=5)
data['dbscan_cluster'] = dbscan.fit_predict(X_scaled)

average_spending = data.groupby('dbscan_cluster')['Spending Score (1-100)'].mean().reset_index()

average_spending.columns = ['cluster', 'average_spending']

print(average_spending)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='dbscan_cluster', palette='viridis')
plt.title('DBSCAN Clustering of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()