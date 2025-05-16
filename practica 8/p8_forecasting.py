import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

df = pd.read_csv("candySalesCleaned.csv", parse_dates=["order_date"])
df['order_date'] = pd.to_datetime(df['order_date'])
df['month'] = df['order_date'].dt.to_period('M')
monthly_sales = df.groupby('month')['sales'].sum().reset_index()
monthly_sales['month'] = monthly_sales['month'].dt.to_timestamp()
monthly_sales['time_index'] = np.arange(len(monthly_sales))
X = monthly_sales[['time_index']]   # independiente (meses como número)
y = monthly_sales['sales']          # dependiente (ventas)
# Entrenar modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"🔍 R² del modelo de regresión lineal: {r2:.4f}")
# Graficar
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales['month'], y, label="Ventas reales", marker='o')
plt.plot(monthly_sales['month'], y_pred, label="Predicción lineal", linestyle='--', color='red')
plt.title("📈 Ventas mensuales con predicción lineal")
plt.xlabel("Mes")
plt.ylabel("Ventas ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Prediccion
futuro = pd.DataFrame({'time_index': [len(monthly_sales)]})
pred_futuro = model.predict(futuro)[0]
print(f"📅 Predicción de ventas para el próximo mes: ${pred_futuro:.2f}")
