import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("candySalesCleaned.csv")
df = df[['sales', 'region']].dropna()
regions = df['region'].unique()
print("Regiones encontradas:", regions)
grouped_sales = [df[df['region'] == region]['sales'] for region in regions]

# == ANOVA ==
anova_result = stats.f_oneway(*grouped_sales)
print("\nANOVA:")
print("F-statistic:", anova_result.statistic)
print("p-value:", anova_result.pvalue)
if anova_result.pvalue < 0.05:
    print("Diferencias significativas entre regiones (ANOVA)")
else:
    print("No hay diferencias significativas entre regiones (ANOVA)")

# == Kruskal-Wallis ==
kruskal_result = stats.kruskal(*grouped_sales)
print("\nKruskal-Wallis:")
print("H-statistic:", kruskal_result.statistic)
print("p-value:", kruskal_result.pvalue)
if kruskal_result.pvalue < 0.05:
    print("Diferencias significativas entre regiones (Kruskal-Wallis)")
else:
    print("No hay diferencias significativas entre regiones (Kruskal-Wallis)")

# == Gráfico de apoyo ==
plt.figure(figsize=(10, 6))
sns.boxplot(x='region', y='sales', data=df)
plt.title("Boxplot de Ventas por Región")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("boxplot_region_stat_test.png")
plt.close()
print("\nGráfico guardado como 'boxplot_region_stat_test.png'")
