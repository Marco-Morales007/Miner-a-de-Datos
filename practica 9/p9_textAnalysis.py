import pandas as pd
from wordcloud import WordCloud 
import matplotlib.pyplot as plt

df = pd.read_csv("candySalesCleaned.csv")

text = " ".join(df['product_name'].dropna().astype(str))

wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(" Word Cloud By Marco :)")
plt.show()
