import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Cargar datos
df = pd.read_csv("candySalesCleaned.csv")

# Crear columna binaria para clasificar ventas
threshold = df['sales'].mean()
df['high_sale'] = (df['sales'] > threshold).astype(int)

# Variables predictoras
X = df[['units', 'gross_profit']].fillna(0)
y = df['high_sale']

# Normalizar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Entrenar modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predicciones
y_pred = knn.predict(X_test)

# Resultados
print("📊 Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\n📄 Reporte de Clasificación:")
print(classification_report(y_test, y_pred))
