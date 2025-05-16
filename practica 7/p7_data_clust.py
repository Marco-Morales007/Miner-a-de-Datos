import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv("candySalesCleaned.csv")

# Usamos solo columnas numéricas relevantes
X = df[['sales', 'units', 'gross_profit', 'cost']]

# Escalamos los datos para mejorar el rendimiento de KMeans
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encontrar número óptimo de clusters con método del codo
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Gráfico del método del codo
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia")
plt.title("Método del Codo para determinar k")
plt.grid(True)
plt.show()

# Elegimos k = 3 por ejemplo (ajustar según el codo)
kmeans = KMeans(n_clusters=3, random_state=0)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualización de los clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sales', y='gross_profit', hue='cluster', palette='Set2')
plt.title("Clusters de Ventas por Ganancia Bruta")
plt.xlabel("Ventas")
plt.ylabel("Ganancia bruta")
plt.grid(True)
plt.show()

# Mostrar cuántos hay por cluster
print("\nDistribución por cluster:")
print(df['cluster'].value_counts())
