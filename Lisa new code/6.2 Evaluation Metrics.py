import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from scipy.stats import f_oneway, pearsonr, spearmanr

# Datensatz laden

data = pd.read_csv("../../Data/BigCitiesHealth.csv")

# Relevante Gesundheitsmetriken auswählen
relevant_metrics = ['Diabetes Deaths', 'Life Expectancy', 'All Cancer Deaths', 'Breast Cancer Deaths', 
                    'Lung Cancer Deaths', 'Cardiovascular Disease Deaths', 'Heart Disease Deaths', 
                    'High Blood Pressure', 'Diabetes', 'Pneumonia or Influenza Deaths', 'Maternal Deaths', 
                    'Infant Deaths', 'Low Birthweight', 'Adult Mental Distress', 'Teen Mental Distress']
features = ['strata_race_label', 'strata_sex_label', 'geo_strata_poverty', 'geo_strata_Segregation', 'geo_strata_region', 'geo_strata_PopDensity']
metrics = relevant_metrics


# Daten filtern
filtered_data = data[data['metric_item_label'].isin(relevant_metrics)]
pivot_data = filtered_data.pivot_table(
    index=['geo_label_city', 'geo_label_state'],
    columns='metric_item_label',
    values='value',
    aggfunc='mean'
).dropna()

# Korrelation Heatmap
plt.figure(figsize=(14, 22))
correlation_matrix = pivot_data.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, linewidths=0.5)
plt.title("Correlation Heatmap of Relevant Health Metrics", fontsize=16, fontweight='bold')
plt.savefig("../Output/6. Heatmap Correlation.jpg")
plt.show()

X = pivot_data
y = pivot_data['Life Expectancy']  # Beispiel für Overall Health

# Modellierung mit RandomForest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
rf_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
print(f"RandomForest Cross-validated R²: {np.mean(rf_scores):.3f}")
plt.figure(figsize=(18, 12))
rf_importances.sort_values().plot(kind='barh')
plt.title('Feature Importance (RandomForest)')
#plt.savefig("../Outputs/6. Metric Analysis - Random Forest.jpg")
plt.show()


# Feature Importance Plot - xgb
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X, y)
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
xgb_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')
print(f"XGBoost Cross-validated R²: {np.mean(xgb_scores):.3f}")

plt.figure(figsize=(18, 12))
xgb_importances.sort_values().plot(kind='barh')
plt.title('Feature Importance (XGBoost)')
#plt.savefig("../Outputs/6. Metric Analysis - GXBoost.jpg")
plt.show()


# Modellierung mit Linear Regression
lr_model = LinearRegression()
lr_scores = cross_val_score(lr_model, X, y, cv=5, scoring='r2')
print(f"Linear Regression Cross-validated R²: {np.mean(lr_scores):.3f}")


# Linear Regression Modell
lr_model = LinearRegression()
lr_model.fit(X, y)
lr_importances = pd.Series(np.abs(lr_model.coef_), index=X.columns)

plt.figure(figsize=(18, 12))
lr_importances.sort_values().plot(kind='barh')
plt.title('Feature Importance (Linear Regression)')
#plt.savefig("../Output/6. Metric Analysis - Linear Regression.jpg")
plt.show()


# Summiere die Importances
combined_importances = rf_importances.add(xgb_importances, fill_value=0).sort_values(ascending=False)
print(combined_importances)
# Feature Importance Plot
plt.figure(figsize=(18, 12))
combined_importances.sort_values().plot(kind='barh')
plt.title('Combined Feature Importance (RandomForest + XGBoost)')
#plt.savefig("../Output/6. Overall Metric Analysis.jpg")
plt.show()


# Correlation through pivot table
for metric in metrics:
    filtered_data = data[data['metric_item_label'].isin([metric, 'Life Expectancy'])]

    # Pivot über Städte
    pivot_data = filtered_data.pivot_table(
        index='geo_label_city',
        columns='metric_item_label',
        values='value',
        aggfunc='mean'
    ).dropna()

    # Überprüfen, ob Pivot-Tabelle Werte enthält
    if pivot_data.empty:
        print(f"Fehler: Die Pivot-Tabelle ist leer für Metrik '{metric}'.")
    else:
        target = pivot_data[metric]
        life_expectancy = pivot_data['Life Expectancy']

        if len(target) > 1 and len(life_expectancy) > 1:
            # ANOVA (über binned Features)
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

            # Pearson und Spearman
            pearson_corr, _ = pearsonr(target, life_expectancy)
            spearman_corr, _ = spearmanr(target, life_expectancy)

            print(f"\nSignifikanzanalyse für '{metric}':")
            print(f"  Pearson-Korrelation mit Life Expectancy: {pearson_corr:.3f}")
            print(f"  Spearman-Korrelation mit Life Expectancy: {spearman_corr:.3f}")

            # Optional: Visualisierung
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=life_expectancy, y=target)
            plt.xlabel("Life Expectancy")
            plt.ylabel(metric)
            plt.title(f"Korrelation: Life Expectancy vs. {metric}")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            stat_df = pd.DataFrame(stat_results).T
            print(stat_df)

            # Visualisierung der ANOVA-p-Werte (Heatmap)
            if not stat_df.empty:
                plt.figure(figsize=(8, 4))
                sns.heatmap(stat_df.T, cmap='coolwarm', annot=True)
                plt.title(f"ANOVA Significance for '{metric}'")
                plt.tight_layout()
                plt.show()


