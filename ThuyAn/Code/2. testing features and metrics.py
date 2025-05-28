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
targets = ['Cardiovascular Disease Deaths', 'Diabetes Deaths', 'Injury Deaths', 'All Cancer Deaths']
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

for target in targets:
    print(f"\nProcessing Metric: {target}")
    df_metric = data[data['metric_item_label'] == target].dropna(subset=features + ['value'])

    # ANOVA für alle Features
    all_features = [col for col in data.columns if col not in ['value', 'metric_item_label']]

    for feature in all_features:
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

# Gesamtergebnisse (über alle Targets)
for feature in all_features:
    groups = [data[data[feature] == val]['value'].dropna().values for val in data[feature].unique() if len(data[data[feature] == val]) > 1]
    if len(groups) > 1:
        f_stat, p_value = f_oneway(*groups)
        significance = "Yes" if p_value < 0.05 else "No"
        anova_results_all_features.append({
            'Metric': 'All Metrics',
            'Feature': feature,
            'F-Statistic': round(f_stat, 4),
            'p-value': round(p_value, 4),
            'Significant': significance
        })

# Ergebnisse in einem DataFrame speichern
anova_df_per_target = pd.DataFrame(anova_results_per_target)
anova_df_all_features = pd.DataFrame(anova_results_all_features)

# Sicherstellen, dass der Pfad existiert
output_path = os.path.join(os.getcwd(), "../Outputs/2. ANOVA_Results_Comparison.xlsx")

with pd.ExcelWriter(output_path) as writer:
    anova_df_per_target.to_excel(writer, sheet_name="ANOVA_Per_Target", index=False)
    anova_df_all_features.to_excel(writer, sheet_name="ANOVA_All_Metrics", index=False)

print(f"\nANOVA-Ergebnisse erfolgreich in {output_path} gespeichert.")


# Identifikation von immer signifikanten Features
significant_features = set(all_features)
for target in targets:
    significant_in_target = set(anova_df_per_target[(anova_df_per_target['Metric'] == target) & (anova_df_per_target['Significant'] == 'Yes')]['Feature'])
    significant_features &= significant_in_target

print("Final Feauture Selection: ", significant_features)
