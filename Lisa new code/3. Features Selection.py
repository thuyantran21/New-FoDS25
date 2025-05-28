import pandas as pd
from scipy.stats import f_oneway
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Laden der Daten
data = pd.read_csv("../Data/BigCitiesHealth.csv")

# Definiere die Zielmetriken und demographischen Merkmale
metrics = data["metric_item_label"].unique()
print("All metrics: ", metrics)

# Metrics - Results from 2. Metric Evaluation
targets = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths', 
           'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']

features = [
    'strata_race_label',
    'strata_sex_label',
    'geo_strata_poverty',
    'geo_strata_Segregation',
    'geo_strata_region',
    'geo_strata_PopDensity',
    'geo_strata_Population'
]

# Speichere ANOVA-Ergebnisse für alle Features
anova_results_all_features = []
anova_results_per_target = []
sign_feature = []

# Zählt, wie oft jedes Feature signifikant ist
feature_significance_count = {feature: 0 for feature in features}

for target in targets:
    print(f"\nProcessing Metric: {target}")
    df_metric = data[data['metric_item_label'] == target].dropna(subset=features + ['value'])

    # ANOVA für alle Features
    for feature in features:
        groups = [group['value'].values for _, group in df_metric.groupby(feature) if len(group) > 1]
        if len(groups) > 1:
            f_stat, p_value = f_oneway(*groups)
            significance = "Yes" if p_value < 0.05 else "No"
            anova_results_per_target.append({
                'Metric': target,
                'Feature': feature,
                'F-Statistic': round(f_stat, 4),
                'p-value': round(p_value, 4),
                'Significant': significance
            })
            
            # Zählt Signifikanz für jedes Feature
            if p_value < 0.05:
                feature_significance_count[feature] += 1



# Ergebnisse in einem DataFrame speichern
anova_df_per_target = pd.DataFrame(anova_results_per_target)

# Sicherstellen, dass der Pfad existiert
output_path = os.path.join(os.getcwd(), "../Output/3. ANOVA_Results_Comparison.xlsx")

with pd.ExcelWriter(output_path) as writer:
    anova_df_per_target.to_excel(writer, sheet_name="ANOVA_Per_Target", index=False)

#print(f"\nANOVA-Ergebnisse erfolgreich in {output_path} gespeichert.")


# Lade die ANOVA-Ergebnistabelle aus der hochgeladenen Excel-Datei
anova_df = pd.read_excel(output_path, sheet_name="ANOVA_Per_Target")

# Identifiziere die Features, die in allen Zeilen "Yes" für "Significant" haben
features = []


# Überprüfe für jedes Feature, ob es maximal ein "No" für "Significant" hat
for feature in anova_df['Feature'].unique():
    no_count = anova_df[(anova_df['Feature'] == feature) & (anova_df['Significant'] == 'No')].shape[0]
    
    # Nur hinzufügen, wenn es maximal ein "No" gibt
    if no_count <= 1:
        features.append(feature)

# Ausgabe
print(f"\nFinal Feature Selection: \n{features}")
