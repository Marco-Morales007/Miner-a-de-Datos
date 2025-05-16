import pandas as pd #pip install pandas
import seaborn as sns #pip install seaborn
import matplotlib.pyplot as plt #python -m pip install -U pip python -m pip install -U matplotlib
from sklearn.linear_model import LinearRegression #pip install scikit-learn
from sklearn.metrics import r2_score

df = pd.read_csv("candySalesCleaned.csv")

df_model = df[['units', 'sales']].dropna()

X = df_model[['units']]
y = df_model['sales']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
print(f"\nCoeficiente de determinación R²: {r2:.4f}")

print(f"Pendiente (coef_): {model.coef_[0]:.4f}")
print(f"Intercepto: {model.intercept_:.4f}")

plt.figure(figsize=(8, 5))
sns.scatterplot(x='units', y='sales', data=df_model, label='Datos reales')
plt.plot(df_model['units'], y_pred, color='red', label='Modelo lineal')
plt.title("Modelo Lineal: Ventas vs Unidades")
plt.xlabel("Unidades Vendidas")
plt.ylabel("Ventas ($)")
plt.legend()
plt.tight_layout()
plt.savefig("linear_model_units_sales.png")
plt.close()
print("Gráfico guardado como 'linear_model_units_sales.png'")
