import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- SETTINGS ----------------
data = pd.read_csv("../Data/BigCitiesHealth_Cleaned.csv")

targets = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths', 'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']

# Zähle Datenpunkte pro Metrik

metric_counts = (
    data[data['metric_item_label'].isin(targets)]
    .groupby('metric_item_label')
    .size()
    .sort_values(ascending=False)
)

total_count = metric_counts.sum()

# DataFrame für Plot
df_counts = metric_counts.reset_index()
df_counts.columns = ['Metric', 'Count']
df_counts['Proportion'] = df_counts['Count'] / total_count

# ---------------- PLOT ----------------
plt.figure(figsize=(10, 6))
sns.barplot(data=df_counts, x="Metric", y="Count", color="skyblue")

plt.title("Data Distribution of Relevant Health Metrics", fontsize=16)
plt.ylabel("Data Amount")
plt.xlabel("Health Metric")
plt.xticks(rotation=45, ha='right')
plt.ylim(0, total_count * 1.05)

# Optionale Beschriftung
for i, row in df_counts.iterrows():
    plt.text(i, row['Count'] + total_count * 0.01, f"{int(row['Count'])}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("../Output/8.5 Metric_Data_Distribution.png")
plt.show()
