import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, make_scorer, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset (cleaned)
data = pd.read_csv("../../Data/BigCitiesHealth_Cleaned.csv")


# Define target metrics (outcome variables) and features    --> Results from Metric and Feature Selection
metrics = ['Lung Cancer Deaths', 'Adult Mental Distress', 
           'Life Expectancy', 'High Blood Pressure', 'Maternal Deaths']
features = ['strata_race_label','strata_sex_label','geo_strata_poverty','geo_strata_Segregation','geo_strata_region','geo_strata_PopDensity']


# Store results
model_results_classification = {}
model_results_regression = {}
feature_importances = {}

# Visualization Setup - Separate Plots for better visibility
combined_importance = {feature: 0 for feature in features}
fig1, axes1 = plt.subplots(1, 5, figsize=(20, 6))  # Feature Importance
fig2, axes2 = plt.subplots(1, 5, figsize=(20, 6))  # Feature Distribution
# fig3, axes3 = plt.subplots(7, 5, figsize=(12, 20))  # Regression Performance
val = []
for i, metric in enumerate(metrics):
    print(f"\nProcessing Metric: {metric}")
    data_metric = data[data['metric_item_label'] == metric].dropna(subset=features + ['value']).copy()

    for feature in features:
        le = LabelEncoder()
        data_metric[feature] = le.fit_transform(data_metric[feature])

    X = data_metric[features]
    y_classification = (data_metric['value'] > data_metric['value'].median()).astype(int)

    # Classification Model
    clf = XGBClassifier(random_state=42)
    clf.fit(X, y_classification)
    importance = dict(zip(features, clf.feature_importances_))
    feature_importances[metric] = importance
    
    # ➕ Ausgabe der Zahlenwerte im Terminal
    print(f"\n📊 Feature Importance für '{metric}':")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat, val in sorted_imp:
        print(f"  - {feat}: {val:.4f}")

    # Aufsummieren
    for f, v in importance.items():
        combined_importance[f] += v
    # Feature Importance Plot
    axes1[i].bar(range(len(importance)), list(importance.values()))
    axes1[i].set_xticks(range(len(importance)))
    axes1[i].set_xticklabels(list(importance.keys()), rotation=45, ha='right', fontsize=9)
    axes1[i].set_title(f"Feature Importance - {metric}")

    # Feature Distribution Plot
    for feature in features:
        sns.histplot(data_metric[feature], ax=axes2[i], kde=True, label=feature, alpha=0.5)
    axes2[i].legend()
    axes2[i].set_title(f"Feature Distribution - {metric}")

fig_overall, ax_overall = plt.subplots(figsize=(8, 6))

# Optional: sortieren für bessere Übersicht
sorted_combined = dict(sorted(combined_importance.items(), key=lambda x: x[1], reverse=True))
ax_overall.bar(sorted_combined.keys(), sorted_combined.values())
ax_overall.set_xticklabels(sorted_combined.keys(), rotation=45, ha='right')
ax_overall.set_title("Feature Importance in Overall Health")
ax_overall.set_ylabel("Overall Sum of Importance (for all metrics)")
fig_overall.tight_layout()
fig_overall.savefig("../Outputs/7.2 Feature_Importance_Overall_Health.pdf")
plt.show()    

# Layout & Save
fig1.tight_layout(pad=2.0)
fig2.tight_layout()
fig1.savefig("../Outputs/7.2 Feature_Importance.pdf")
fig2.savefig("../Outputs/7.2 Feature_Distribution.pdf")

plt.show()


