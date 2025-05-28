import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os

# Laden der Daten
data = pd.read_csv('BigCitiesHealth.csv')

# Features und Zielvariablen (Targets)
features = [
    'strata_race_label',
    'strata_sex_label',
    'geo_strata_poverty',
    'geo_strata_region',
    'geo_strata_PopDensity'
]

metrics = [
    'Cardiovascular Disease Deaths',
    'Diabetes Deaths',
    'Injury Deaths',
    'All Cancer Deaths'
]

# Datenvorverarbeitung
filtered_data = data[data['metric_item_label'].isin(metrics)]
X = filtered_data[features]
y = filtered_data['metric_item_label']



# Subgroup-Plots erstellen
for hue in features:
    fig, axes = plt.subplots(1, len(features), figsize=(25, 5), constrained_layout=True)
    
    for ax, feature in zip(axes, features):
        sns.countplot(x=feature, data=filtered_data, ax=ax, hue=hue, palette='viridis')
        ax.set_title(f'{feature} (Hue = {hue})')

    plt.suptitle(f'Countplots for Features with Hue = {hue}', fontsize=16)
    plt.savefig(f"../Outputs/9.2 Subgroups Distribution - {hue}.jpg")
    plt.show()

# Distribution Vergleichsgrafik
plt.figure(figsize=(12,6))
feature_counts = filtered_data[features].count().sort_values()
sns.barplot(y=feature_counts.index, x=feature_counts.values)
plt.title('Feature Value Distribution - Check')
plt.xlabel('Anzahl der Werte')
plt.ylabel('Features')
plt.savefig('../Outputs/9.2 Feature Value Distribution - Check.jpg')
plt.show()

# Feature Value Distribution
feature_counts = filtered_data[features].count()
if feature_counts.nunique() == 1:
    print("Alle Features haben die gleiche Anzahl an Werten:", feature_counts.iloc[0])
else:
    print("Features haben unterschiedliche Anzahl an Werten:")
    print(feature_counts)

