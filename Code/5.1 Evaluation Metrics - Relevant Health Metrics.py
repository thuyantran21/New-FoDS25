import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from scipy.stats import f_oneway, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

#######################################################################################################
########################################### General Settings ##########################################
#######################################################################################################
# Datensatz laden
data = pd.read_csv("../Data/BigCitiesHealth.csv")

# Relevante Gesundheitsmetriken auswählen
all_metrics = [
    'Diabetes Deaths', 'Cardiovascular Disease Deaths', 'Heart Disease Deaths',
    'High Blood Pressure', 'Adult Obesity', 'Teen Obesity',
    'All Cancer Deaths', 'Breast Cancer Deaths', 'Lung Cancer Deaths',
    'Life Expectancy', 'Deaths from All Causes',
    'Adult Mental Distress', 'Pneumonia or Influenza Deaths',
    'Low Birthweight'
]
#features = ['strata_race_label', 'strata_sex_label', 'geo_strata_poverty', 'geo_strata_Segregation', 'geo_strata_region', 'geo_strata_PopDensity']
relevant_metrics = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths', 'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']

metrics = relevant_metrics
features = [
    'strata_race_label',
    'strata_sex_label',
    'geo_strata_poverty',
    'geo_strata_region',
    'geo_strata_PopDensity'
]
#######################################################################################################
########################################## Data Preparation ###########################################
#######################################################################################################
# Daten filtern und pivotieren
filtered_data = data[data['metric_item_label'].isin(relevant_metrics)]
pivot_data = filtered_data.pivot_table(
    index=['geo_label_city', 'geo_label_state'],
    columns='metric_item_label',
    values='value',
    aggfunc='mean'
).dropna()

X = pivot_data
y = pivot_data['Life Expectancy']

# Standardisiertes X für KNN etc.
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

#######################################################################################################
########################################## Correlation Heatmap ########################################
#######################################################################################################
plt.figure(figsize=(14, 22))
correlation_matrix = pivot_data.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, linewidths=0.5)
plt.title("Correlation Heatmap of Relevant Health Metrics", fontsize=16, fontweight='bold')
# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')  # Rotate labels 45 degrees and align to the right
# Rotate y-axis labels
plt.yticks(rotation=0)  # Rotate labels 0 degrees to keep them horizontal
plt.savefig("../Output/5.1 Heatmap Correlation of Relevant Health Metrics.jpg")
plt.show()

#######################################################################################################
########################################## Metric Importances #########################################
#######################################################################################################
# Modelle + Datenzuordnung
models = {
    "Random Forest": (RandomForestRegressor(random_state=42), X),
    "XGBoost": (XGBRegressor(random_state=42), X),
    "Gradient Boosting": (GradientBoostingRegressor(random_state=42), X),
    "KNN": (KNeighborsRegressor(), X_scaled)
}

# Je nach Modell richtigen X-Datensatz auswählen
X_dict = {
    "Random Forest": X,
    "XGBoost": X,
    "Gradient Boosting": X,
    "KNN": X_scaled
}

# Dictionary zur Speicherung der Importances
feature_importance_dict = {}

# Modelltraining und Bewertung
for name, (model, X_used) in models.items():
    print(f"\n🔍 Modell: {name}")
    try:
        scores = cross_val_score(model, X_used, y, cv=5, scoring='r2')
        print(f"✅ Cross-validated R²: {np.mean(scores):.3f}")
        
        model.fit(X_used, y)

        if hasattr(model, 'feature_importances_'):
            importances = pd.Series(model.feature_importances_, index=X_used.columns)
            feature_importance_dict[name] = importances
            plt.figure(figsize=(12, 8))
            importances.sort_values().plot(kind='barh')
            plt.title(f"Feature Importance: {name}")
            plt.tight_layout()
            plt.savefig(f"../Output/5.1 Relevant Health Metrics Analysis - {name}.jpg")
            plt.show()

        elif name == "KNN":
            result = permutation_importance(model, X_used, y, scoring='r2', n_repeats=10, random_state=42)
            importances = pd.Series(result.importances_mean, index=X_used.columns)
            feature_importance_dict[name] = importances
            plt.figure(figsize=(12, 8))
            importances.sort_values().plot(kind='barh')
            plt.title("Feature Importance (KNN - Permutation)")
            plt.tight_layout()
            plt.savefig("../Output/5.1 Relevant Health Metric Analysis - KNN.jpg")
            plt.show()

    except Exception as e:
        print(f"⚠️ Fehler bei {name}: {e}")

# ----------------------------- Kombinierter Plot -----------------------------


# Gemeinsame Feature-Reihenfolge festlegen (z.B. basierend auf erstem Modell)
base_index = next(iter(feature_importance_dict.values())).index

# Alle Importances korrekt reindizieren und aufsummieren
combined_importances = sum(
    importance.reindex(base_index, fill_value=0)
    for importance in feature_importance_dict.values()
)


# Plot
plt.figure(figsize=(14, 10))
combined_importances.sort_values().plot(kind='barh')
plt.title("Combined Feature Importance (All Models)", fontsize=14, fontweight='bold')
plt.xlabel("Summed Importance Score")
plt.tight_layout()
plt.savefig("../Output/5.1 Combined Feature Importance.jpg")
plt.show()

#######################################################################################################
########################################## Significance Analysis ######################################
#######################################################################################################
for metric in metrics:
    filtered_data = data[data['metric_item_label'].isin([metric, 'Life Expectancy'])]

    pivot_data = filtered_data.pivot_table(
        index='geo_label_city',
        columns='metric_item_label',
        values='value',
        aggfunc='mean'
    ).dropna()

    if pivot_data.empty:
        print(f"⚠️ Fehler: Pivot-Tabelle leer für '{metric}'.")
    else:
        target = pivot_data[metric]
        life_expectancy = pivot_data['Life Expectancy']

        if len(target) > 1 and len(life_expectancy) > 1:
            stat_results = {}
            for f in features:
                if f in data.columns:
                    merged = data[data['metric_item_label'] == metric][[f, 'value']].dropna()
                    if merged[f].nunique() > 1:
                        groups = [group['value'].values for _, group in merged.groupby(f)]
                        f_stat, p_value_anova = f_oneway(*groups)
                    else:
                        p_value_anova = np.nan
                    stat_results[f] = {'ANOVA p-value': p_value_anova}

            pearson_corr, _ = pearsonr(target, life_expectancy)
            spearman_corr, _ = spearmanr(target, life_expectancy)

            print(f"\n📊 Signifikanzanalyse für '{metric}':")
            print(f"  Pearson-Korrelation: {pearson_corr:.3f}")
            print(f"  Spearman-Korrelation: {spearman_corr:.3f}")

            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=life_expectancy, y=target)
            plt.xlabel("Life Expectancy")
            plt.ylabel(metric)
            plt.title(f"Scatter: Life Expectancy vs. {metric}")
            plt.grid(True)
            plt.tight_layout()
            #plt.show()

            stat_df = pd.DataFrame(stat_results).T
            print(stat_df)

            if not stat_df.empty:
                plt.figure(figsize=(8, 4))
                sns.heatmap(stat_df.T, cmap='coolwarm', annot=True)
                plt.title(f"ANOVA Significance for '{metric}'")
                plt.tight_layout()
                #plt.show()
