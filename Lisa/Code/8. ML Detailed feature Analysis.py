import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load dataset (cleaned)
data = pd.read_csv("../../Data/BigCitiesHealth_Cleaned.csv")
average_importance = pd.read_excel("../Outputs/6. ML_Results_Optimized.xlsx", sheet_name="Aggregated_Importance", index_col=0)

# Define significant features based on average importance threshold
importance_threshold = 0.01
significant_features = average_importance[average_importance['Average Importance'] >= importance_threshold].index.tolist()
print("Significant Features (based on Average Importance):", significant_features)

# Define target metrics (outcome variables)
metrics = [
    'Cardiovascular Disease Deaths',
    'Diabetes Deaths',
    'Injury Deaths',
    'All Cancer Deaths'
]

# Define selected features (from Feature Selection Report)
selected_features = ['strata_race_label', 'strata_sex_label', 'geo_strata_poverty', 'geo_strata_region', 'geo_strata_PopDensity']

# Define target metrics (outcome variables)
metrics = [
    'Cardiovascular Disease Deaths',
    'Diabetes Deaths',
    'Injury Deaths',
    'All Cancer Deaths'
]
################################################# Selected Feature Analysis #######################################################
# Store results
feature_importances_selected = {}
feature_importances_all = {}

# Visualization Setup
#fig, axes = plt.subplots(len(metrics), 3, figsize=(20, 15))

# Loop through each metric to build and evaluate models
for i, metric in enumerate(metrics):
    print(f"\nProcessing Metric: {metric}")
    data_metric = data[data['metric_item_label'] == metric].dropna(subset=selected_features + ['value']).copy()

    # Convert all object columns to category
    for col in data_metric.select_dtypes(['object']).columns:
        data_metric[col] = data_metric[col].astype('category')

    # Separate X and y
    X_selected = data_metric[selected_features]
    X_all = data_metric.drop(['value', 'metric_item_label'], axis=1)
    y = data_metric['value']

    # Classification Model (Selected Features)
    clf_selected = XGBClassifier(random_state=42, enable_categorical=True)
    clf_selected.fit(X_selected, (y > y.median()).astype(int))
    feature_importances_selected[metric] = dict(zip(selected_features, clf_selected.feature_importances_))

    # Classification Model (All Features)
    clf_all = XGBClassifier(random_state=42, enable_categorical=True)
    clf_all.fit(X_all, (y > y.median()).astype(int))
    feature_importances_all[metric] = dict(zip(X_all.columns, clf_all.feature_importances_))

# Save Results to Excel
results_selected = pd.DataFrame(feature_importances_selected).T
results_all = pd.DataFrame(feature_importances_all).T

with pd.ExcelWriter("../Outputs/8. ML_Results_Feature_Importance_Comparison.xlsx") as writer:
    results_selected.to_excel(writer, sheet_name="Selected_Features")
    results_all.to_excel(writer, sheet_name="All_Features")

print("Results saved to ML_Results_Feature_Importance_Comparison.xlsx")


################################################# All Feature Analysis ############################################################
# Store results
subgroup_results = {}
category_mappings = {}  # Dictionary to store category mappings for each feature

# Loop through each metric to build and evaluate models
for metric in metrics:
    print(f"\nProcessing Metric: {metric}")
    data_metric = data[data['metric_item_label'] == metric].dropna(subset=significant_features + ['value']).copy()

    # Create mappings for categorical features
    for feature in significant_features:
        if data_metric[feature].dtype == 'object':
            unique_categories = data_metric[feature].unique()
            category_mappings[feature] = {category: idx for idx, category in enumerate(unique_categories)}
            # Map categories to numerical values
            data_metric[feature] = data_metric[feature].map(category_mappings[feature])

    for feature in significant_features:
        print(f"\nAnalyzing Subcategories for Feature: {feature}")
        unique_values = data_metric[feature].unique()

        # Analyze each subgroup separately
        for value in unique_values:
            subgroup_data = data_metric[data_metric[feature] == value].copy()
            X = subgroup_data[significant_features]
            y = (subgroup_data['value'] > subgroup_data['value'].median()).astype(int)

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
                original_value = value  # If not categorical

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
            print(f"    {subgroup}: F1-Score = {score}")                                    # Kontrolle: Spielt Menge an Personen einer Kategorie eine Rolle?

# Converting into DataFrame and saving as Excel-Sheat
# df_table = pd.DataFrame(table, columns=["Metric/Feature", "Subgroup", "F1-Score"])
# df_table.to_excel("../Outputs/8. Overview of Subgroup Performance Anaylsis.xlsx")         # Not necessary as already done in selected features part

# Plotting Subplots for each Feature and their Subcategories
# Plotting Subplots for each Feature and their Subcategories
for metric, features in subgroup_results.items():
    num_features = len(features)
    fig, axes = plt.subplots(1, num_features, figsize=(5 * num_features, 5))
    fig.suptitle(f"Subgroup Performance Comparison for {metric}", fontsize=16)


    colors = np.random.rand(len(list(subgroups.values())), 3)

    # Ensure axes is always iterable
    if num_features == 1:
        axes = [axes]

    for ax, (feature, subgroups) in zip(axes, features.items()):
        categories = list(subgroups.keys())
        scores = list(subgroups.values())
        ax.bar(categories, scores, color=colors)
        ax.set_title(feature)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_ylabel("F1-Score")
        ax.set_ylim(0, 1)
        #ax.set_xticklabels(categories, rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0)
    plt.savefig(f"../Outputs/8. {metric}_Subgroup_Performance.png")  # Save each plot
    plt.show()


# Subplot with all metrics together
# Assuming subgroup_results is already defined
fig, axes = plt.subplots(4, 5, figsize=(25, 20))  # 4x5 layout for 4 metrics
fig.suptitle("Subgroup Performance Comparison for All Metrics", fontsize=16)

metrics = list(subgroup_results.keys())

# Define color palettes for each row
color_palettes = [
    ['#2E8B57', '#66CDAA', '#4682B4', '#5F9EA0'],  # Green and blue shades
    ['#FFA500', '#FFD700', '#FF6347', '#FF4500'],  # Yellow and orange shades
    ['#6A5ACD', '#8A2BE2', '#7B68EE', '#4B0082'],  # Violet and blue shades
    ['#DAA520', '#F4A460', '#D2691E', '#CD853F', '#FFD700']  # Yellow and brown shades
]

for i, metric in enumerate(metrics):
    features = subgroup_results[metric]
    for j, (feature, subgroups) in enumerate(features.items()):
        ax = axes[i, j]  # Select the correct subplot
        categories = list(subgroups.keys())
        scores = list(subgroups.values())

        # Cycle through colors for each category in the current row
        colors = [color_palettes[i % 4][k % len(color_palettes[i % 4])] for k in range(len(categories))]

        ax.bar(categories, scores, color=colors)
        ax.set_title(feature)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_ylabel("F1-Score")
        ax.set_ylim(0, 1)

# Adjust layout for clean display
plt.tight_layout(rect=[0, 0, 1, 0.95], w_pad=2.0, h_pad=2.0)
plt.savefig("../Outputs/8. Combined_Subgroup_Performance.png")  # Save the combined plot
plt.show()