import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Beispiel-Daten laden
data = pd.read_csv('BigCitiesHealth_Cleaned.csv')  # Ersetze durch deinen Pfad

# Definiere Features und Zielmetriken
features = [
    'strata_race_label',
    'strata_sex_label',
    'geo_strata_poverty',
    'geo_strata_Segregation',
    'geo_strata_region',
    'geo_strata_PopDensity',
    'geo_strata_Population'
]

metrics = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths', 
           'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']


# Dictionary, um die Ergebnisse zu speichern
subgroup_ratios = []

# Berechnung des Anteils für jede Subgruppe in jeder Metrik
for metric in metrics:
    data_metric = data[data['metric_item_label'] == metric]
    
    for feature in features:
        subgroup_counts = data_metric[feature].value_counts(normalize=True) * 100
        for subgroup, ratio in subgroup_counts.items():
            subgroup_ratios.append({
                'Metric': metric,
                'Feature': feature,
                'Subgroup': subgroup,
                'Percentage': ratio
            })

# Umwandlung in DataFrame
subgroup_ratios_df = pd.DataFrame(subgroup_ratios)
print("\nAnteil der Subgruppen an den gesamten Features pro Metric (in %):")
print(subgroup_ratios_df)

# Visualisierung
plt.figure(figsize=(15, 10))
sns.barplot(
    data=subgroup_ratios_df,
    x='Percentage',
    y='Metric',
    hue='Feature',
    orient='h'
)
plt.title("Anteil der Subgruppen an den Zielmetriken")
plt.xlabel("Anteil (%)")
plt.ylabel("Metric")
plt.tight_layout()
plt.savefig("../Output/8.3 Subgroup Distribution by Metrics.png")
#plt.show()


# Dictionary, um die Ergebnisse zu speichern
subgroup_ratios = []

# Berechnung des Anteils für jede Subgruppe in jeder Metrik
for metric in metrics:
    data_metric = data[data['metric_item_label'] == metric]
    
    for feature in features:
        subgroup_counts = data_metric[feature].value_counts(normalize=True) * 100
        for subgroup, ratio in subgroup_counts.items():
            subgroup_ratios.append({
                'Metric': metric,
                'Feature': feature,
                'Subgroup': subgroup,
                'Percentage': ratio
            })

# Umwandlung in DataFrame
subgroup_ratios_df = pd.DataFrame(subgroup_ratios)
print("\nAnteil der Subgruppen an den gesamten Features pro Metric (in %):")
print(subgroup_ratios_df)

# Visualisierung: 7 Plots (1 für jede Metrik) mit jeweils 2x3 Subplots (6 Features)
for metric in metrics:
    metric_data = subgroup_ratios_df[subgroup_ratios_df['Metric'] == metric]
    
    # Erstelle eine 2x3-Subplot-Struktur für die 6 Features
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(f"Subgroup Distribution for {metric}", fontsize=16)

    # Iteriere über Features und fülle die Subplots
    for ax, feature in zip(axes.flat, features):
        feature_data = metric_data[metric_data['Feature'] == feature]
        sns.barplot(
            data=feature_data,
            x='Subgroup',
            y='Percentage',
            hue='Subgroup',
            ax=ax
        )
        ax.set_title(feature)
        ax.set_xlabel("Subgroup")
        ax.set_ylabel("Anteil (%)")
        ax.tick_params(axis='x', rotation=45)
    
    # Gemeinsame Legende für alle Subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=12)
    
    plt.savefig(f"../Output/8.3 Subgroup Distribution - {metric}.png")
    #plt.show()
