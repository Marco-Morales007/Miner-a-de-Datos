import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("candySalesCleaned.csv")
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
df['ship_date'] = pd.to_datetime(df['ship_date'], errors='coerce')
# Columnas para año y mes
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month
# --- Estadísticas generales ---
print("\n== Estadísticas Generales ==")
print(df[['units', 'sales', 'cost', 'gross_profit']].describe())
# --- Ventas por producto ---
ventas_por_producto = df.groupby('product_id')[['units', 'sales', 'gross_profit']].sum().sort_values(by='sales', ascending=False)
print("\n== Ventas por Producto ==")
print(ventas_por_producto)
# --- Ventas por cliente ---
ventas_cliente = df.groupby('customer_id')[['sales']].agg(['sum', 'mean', 'count'])
print("\n== Ventas por Cliente ==")
print(ventas_cliente)
# --- Ventas por región ---
ventas_region = df.groupby('region')['sales'].sum().sort_values(ascending=False)
print("\n== Ventas por Región ==")
print(ventas_region)
# --- Ventas por año ---
ventas_anuales = df.groupby('year')['sales'].sum()
print("\n== Ventas por Año ==")
print(ventas_anuales)

# GRAFICA
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x=ventas_anuales.index, y=ventas_anuales.values)
plt.title("Ventas Totales por Año")
plt.ylabel("Ventas")
plt.xlabel("Año")
plt.tight_layout()
plt.show()
