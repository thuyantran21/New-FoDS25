import pandas as pd
import matplotlib.pyplot as plt

# Beispiel-Daten laden
data = pd.read_csv("../Data/BigCitiesHealth_Cleaned.csv")  # Ersetze durch deinen Pfad

# Die Features, für die der Anteil der Subgruppen berechnet wird
features = [
    'strata_race_label',
    'strata_sex_label',
    'geo_strata_poverty',
    'geo_strata_Segregation',
    'geo_strata_region',
    'geo_strata_PopDensity'
]

# Dictionary, um die Ergebnisse zu speichern
subgroup_ratios = {}

# Berechnung des Anteils für jede Subgruppe
for feature in features:
    subgroup_counts = data[feature].value_counts(normalize=True) * 100
    subgroup_ratios[feature] = subgroup_counts

# Visualisierung: 2x3 Subplot-Layout
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flat  # Konvertiere zu flacher Liste für einfacheren Zugriff

for ax, feature in zip(axes, features):
    # Überprüfe, ob das Feature in den berechneten Subgruppen vorhanden ist
    if feature in subgroup_ratios:
        subgroup_ratios[feature].plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(f'Anteil der Subgruppen - {feature}')
        #ax.set_xlabel('Subgruppen')
        ax.set_ylabel('Anteil (%)')
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.set_visible(False)  # Verstecke den Subplot, wenn keine Subgruppe vorhanden

# Entferne leere Subplots (falls weniger als 6 Features)
for ax in axes[len(features):]:
    ax.remove()

plt.tight_layout(h_pad=1.0)
plt.savefig("../Output/8.1 Features Subgroup Distribution.jpg")
plt.show()

