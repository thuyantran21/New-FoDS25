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
data = pd.read_csv("../Data/BigCitiesHealth_Cleaned.csv")

# Features und Zielvariablen (Targets)
features = [
    'strata_race_label',
    'strata_sex_label',
    'geo_strata_poverty',
    'geo_strata_Segregation',
    'geo_strata_region',
    'geo_strata_PopDensity'
]

#metrics = ['Cardiovascular Disease Deaths','Diabetes Deaths','Injury Deaths','All Cancer Deaths']
metrics =  ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths', 
           'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']


# Datenvorverarbeitung
filtered_data = data[data['metric_item_label'].isin(metrics)]
X = filtered_data[features]
y = filtered_data['metric_item_label']



# Dictionary for custom made Labels
custom_labels = {
    'geo_strata_poverty': {
        'Poorest cities (20%+ poor)': 'Yes',
        'Less poor cities (<20% poor)': 'No'
    },
    'geo_strata_PopDensity': {
        'Highest pop. density (>10k per sq mi)': 'High Density',
        'Lower pop. density (<10k per sq mi)': 'Low Density'
    }
}

# 5 Plots with 5 Subplots each
for hue in features:
    fig, axes = plt.subplots(1, len(features), figsize=(25, 25))
    
    for ax, feature in zip(axes, features):
        sns.countplot(x=feature, data=filtered_data, ax=ax, hue=hue, palette='viridis')
        
        # Benutzerdefinierte Labels anwenden (falls definiert)
        if feature in custom_labels:
            ax.set_xticklabels([custom_labels[feature].get(label.get_text(), label.get_text()) 
                                for label in ax.get_xticklabels(10)])
        
        ax.set_title(f'{feature} (Hue = {hue})')
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_ylim(0, 20000) 
        plt.tight_layout(pad=2.0) 
        plt.suptitle(f'Countplots for Features with Hue = {hue}', fontsize=16)
        plt.savefig(f"../Output/8.2 Subgroup Distribution - {feature}.jpg")
    #plt.show()
              

# Distribution Vergleichsgrafik
plt.figure(figsize=(12,6))
feature_counts = filtered_data[features].count().sort_values()
sns.barplot(y=feature_counts.index, x=feature_counts.values)
plt.title('Feature Value Distribution - Check')
plt.xlabel('Value')
plt.ylabel('Features')
plt.savefig('../Output/8.2 Feature Value Distribution - Check.jpg')
#plt.show()

# Feature Value Distribution
feature_counts = filtered_data[features].count()
if feature_counts.nunique() == 1:
    print("All Features have the same amount of Data:", feature_counts.iloc[0])
else:
    print("Features show different amount of Data:")
    print(feature_counts)
