import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("candySalesCleaned.csv")

df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

#carpeta opcional para guardar imágenes
import os
os.makedirs("graficas", exist_ok=True)

# 1. PIE CHART: Top 5 estados con más ventas
state_sales = df.groupby('stateprovince')['sales'].sum().sort_values(ascending=False)
top_states = state_sales.head(5)

plt.figure(figsize=(7, 7))
plt.pie(top_states, labels=top_states.index, autopct='%1.1f%%', startangle=90)
plt.title("Top 5 Estados por Ventas")
plt.savefig("graficas/pie_top_states.png")
plt.close()

# 2. HISTOGRAMAS: Ventas, Costos y Ganancia Bruta
cols = ['sales', 'cost', 'gross_profit']
for col in cols:
    plt.figure()
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Histograma de {col.capitalize()}")
    plt.savefig(f"graficas/hist_{col}.png")
    plt.close()

# 3. BOXPLOTS: Comparación de ventas por región
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='region', y='sales')
plt.xticks(rotation=45)
plt.title("Boxplot de Ventas por Región")
plt.tight_layout()
plt.savefig("graficas/boxplot_region_sales.png")
plt.close()

# 4. SCATTER PLOTS: Ventas vs Costo, Ventas vs Ganancia
scatter_pairs = [('cost', 'sales'), ('gross_profit', 'sales')]
for x, y in scatter_pairs:
    plt.figure()
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f"{y} vs {x}")
    plt.savefig(f"graficas/scatter_{x}_{y}.png")
    plt.close()

# 5. PLOT DE LÍNEA: Ventas por Año
df['year'] = df['order_date'].dt.year
sales_year = df.groupby('year')['sales'].sum()

plt.figure()
sales_year.plot(marker='o')
plt.title("Ventas Totales por Año")
plt.xlabel("Año")
plt.ylabel("Ventas")
plt.grid(True)
plt.savefig("graficas/linea_ventas_anuales.png")
plt.close()

print("Gráficos generados y guardados en la carpeta 'graficas'.")
