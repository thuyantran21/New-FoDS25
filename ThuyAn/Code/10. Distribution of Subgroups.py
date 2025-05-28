import pandas as pd
import matplotlib.pyplot as plt

# Beispiel-Daten laden
data = pd.read_csv('../../Data/BigCitiesHealth_Cleaned.csv')  # Ersetze durch deinen Pfad

# Die Features, für die der Anteil der Subgruppen berechnet wird
features = [
    'strata_race_label',
    'strata_sex_label',
    'geo_strata_poverty',
    'geo_strata_Segregation',
    'geo_strata_region',
    'geo_strata_PopDensity',
    'geo_strata_Population'
]

# Dictionary, um die Ergebnisse zu speichern
subgroup_ratios = {}

# Berechnung des Anteils für jede Subgruppe
for feature in features:
    subgroup_counts = data[feature].value_counts(normalize=True) * 100
    subgroup_ratios[feature] = subgroup_counts

# Ausgabe der Anteile als DataFrame
subgroup_ratios_df = pd.DataFrame(subgroup_ratios).fillna(0)
print("\nAnteil der Subgruppen an den gesamten Features (in %):")
print(subgroup_ratios_df)

# Visualisierung
fig, axes = plt.subplots(1, len(features),  figsize=(12, 2 * len(features)), constrained_layout=True)

for ax, feature in zip(axes, features):
    subgroup_ratios[feature].plot(kind='barh', ax=ax)
    ax.set_title(f'Anteil der Subgruppen - {feature}')
    ax.set_xlabel('Anteil (%)')

plt.show()
