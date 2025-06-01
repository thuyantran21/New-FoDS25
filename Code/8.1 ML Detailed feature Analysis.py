import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load dataset (cleaned)
data = pd.read_csv("../Data/BigCitiesHealth_Cleaned.csv")
average_importance = pd.read_excel("../Outputs/7.1 ML_Results_Optimized.xlsx", sheet_name="Aggregated_Importance", index_col=0)

# Define significant features based on average importance threshold
importance_threshold = 0.01
significant_features = average_importance[average_importance['Average Importance'] >= importance_threshold].index.tolist()
print("Significant Features (based on Average Importance):", significant_features)

# Define selected features (from Feature Selection Report)
selected_features = ['strata_race_label', 'strata_sex_label', 'geo_strata_poverty', 'geo_strata_Segregation', 'geo_strata_region', 'geo_strata_PopDensity']

# Define target metrics (outcome variables)
metrics = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths', 'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']

subgroup_results = {}
category_mappings = {}

# Loop through each metric to build and evaluate models
for metric in metrics:
    print(f"\nProcessing Metric: {metric}")
    data_metric = data[data['metric_item_label'] == metric].dropna(subset=significant_features + ['value']).copy()

    # Create mappings for categorical features
    for feature in significant_features:
        if data_metric[feature].dtype == 'object':
            unique_categories = data_metric[feature].unique()
            category_mappings[feature] = {category: idx for idx, category in enumerate(unique_categories)}
            data_metric[feature] = data_metric[feature].map(category_mappings[feature])

    for feature in significant_features:
        print(f"\nAnalyzing Subcategories for Feature: {feature}")
        unique_values = data_metric[feature].unique()

        for value in unique_values:
            subgroup_data = data_metric[data_metric[feature] == value].copy()
            X = subgroup_data[significant_features]
            y = (subgroup_data['value'] > subgroup_data['value'].median()).astype(int)

            # Skip subgroups with no variability in y (only one class)
            if y.nunique() < 2:
                print(f"Skipping Subgroup {feature} = {value} - Only one class present in y.")
                continue

            # Classification Model
            clf = XGBClassifier(random_state=42)
            clf.fit(X, y)
            score = cross_val_score(clf, X, y, cv=5, scoring='f1').mean()

            # Store results
            if metric not in subgroup_results:
                subgroup_results[metric] = {}
            if feature not in subgroup_results[metric]:
                subgroup_results[metric][feature] = {}

            # Convert numerical value back to original category
            if feature in category_mappings:
                original_value = [k for k, v in category_mappings[feature].items() if v == value][0]
            else:
                original_value = value

            subgroup_results[metric][feature][original_value] = round(score, 3)

# Display results
table = []
print("\nSubgroup Performance Analysis:")
for metric, features in subgroup_results.items():
    table.append(["Scores of {metric}", "", ""])
    print(f"\nMetric: {metric}")
    for feature, subgroups in features.items():
        print(f"  Feature: {feature}")
        table.append([feature, "", ""])
        for subgroup, score in subgroups.items():
            table.append(["", subgroup, score])
            print(f"    {subgroup}: F1-Score = {score}")

# Save results to Excel
df_table = pd.DataFrame(table, columns=["Metric/Feature", "Subgroup", "F1-Score"])
df_table.to_excel("../Outputs/8. Subgroup_Performance_Analysis_Optimized.xlsx", index=False)
print("\nResults saved to ../Outputs/8. Subgroup_Performance_Analysis_Optimized.xlsx")
