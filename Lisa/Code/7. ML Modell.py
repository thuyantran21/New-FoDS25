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
data = pd.read_csv("../Data/BigCitiesHealth_Cleaned.csv")

# Define selected features (from Feature Selection Report)
#features = ['strata_race_label',  'strata_sex_label', 'geo_strata_poverty', 'geo_strata_region', 'geo_strata_PopDensity']

features = ['strata_race_label','strata_sex_label','geo_strata_poverty','geo_strata_region','geo_strata_PopDensity']

# Define target metrics (outcome variables)
metrics = [
    'Cardiovascular Disease Deaths',
    'Diabetes Deaths',
    'Injury Deaths',
    'All Cancer Deaths'
]

# Store results
model_results_classification = {}
model_results_regression = {}
feature_importances = {}

# Hyperparameter Grid for RandomizedSearchCV
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'n_estimators': [50, 100, 200, 500],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Visualization Setup
fig, axes = plt.subplots(len(metrics), 3, figsize=(20, 15))
combined_importance = {feature: 0 for feature in features}

# Loop through each metric to build and evaluate models
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

    # Add to combined importance
    for feature, value in importance.items():
        combined_importance[feature] += value

    
    # Feature Importance Plot
    axes[i, 0].bar(importance.keys(), importance.values())
    axes[i, 0].set_title(f"Feature Importance - {metric}")

    # Distribution Plot
    for feature in features:
        sns.histplot(data_metric[feature], ax=axes[i, 1], kde=True, label=feature, alpha=0.5)
    axes[i, 1].legend()
    axes[i, 1].set_title(f"Feature Distribution - {metric}")

    # Regression Performance Plot
    reg = XGBRegressor(random_state=42)
    reg.fit(X, data_metric['value'])
    y_pred = reg.predict(X)
    axes[i, 2].scatter(data_metric['value'], y_pred, alpha=0.5)
    axes[i, 2].set_xlabel("True Values")
    axes[i, 2].set_ylabel("Predicted Values")
    axes[i, 2].set_title(f"Regression Performance - {metric}")
    axes[i, 0].tick_params(axis='x', rotation=45)

plt.tight_layout(pad=2.0)   
plt.savefig("../Outputs/6. ML Modell Results Overview.pdf", format='pdf')
plt.savefig("../Outputs/6. ML Modell Results Overview.jpg", format='jpg')

plt.show()

# Calculate average importance across all metrics
average_importance = {feature: value / len(metrics) for feature, value in combined_importance.items()}

# Plotting Aggregated Feature Importance
plt.figure(figsize=(10, 6))
plt.bar(average_importance.keys(), average_importance.values())
plt.title("Aggregated Feature Importance over all metrics:")
plt.xlabel("Features")
plt.ylabel("Average Importance")
plt.savefig("../Outputs/6. Aggregated_Feature_Importance.pdf", format='pdf')
plt.show()
print("\nAggregated Feature Importance over all metrics:")
for feature, value in average_importance.items():
    print(f"{feature}: {value:.4f}")

# Save Results to Excel
results_df_classification = pd.DataFrame(model_results_classification).T
results_df_regression = pd.DataFrame(model_results_regression).T
importance_df = pd.DataFrame(feature_importances)

average_importance_df = pd.DataFrame.from_dict(average_importance, orient='index', columns=['Average Importance'])

with pd.ExcelWriter("../Outputs/6. ML_Results_Optimized.xlsx") as writer:
    results_df_classification.to_excel(writer, sheet_name="Classification_Results")
    results_df_regression.to_excel(writer, sheet_name="Regression_Results")
    importance_df.to_excel(writer, sheet_name="Feature_Importance")
    average_importance_df.to_excel(writer, sheet_name="Aggregated_Importance")

print("Results saved to ML_Results_Optimized.xlsx")
