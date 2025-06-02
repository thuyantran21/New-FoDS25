import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv("../Data/BigCitiesHealth_Cleaned.csv")
average_importance = pd.read_excel("../Output/6.1_Model_Importances.xlsx", index_col=0)

# Select significant features based on importance threshold
importance_threshold = 0.01
significant_features = average_importance.loc[average_importance['Importance'] >= importance_threshold].iloc[:, 1].unique().tolist()
print("Significant Features (based on Importance):", significant_features)

# Define target metrics
metrics = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths',
           'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']

subgroup_results = {}
category_mappings = {}

# Evaluate models per metric and subgroup
for metric in metrics:
    print(f"\nProcessing Metric: {metric}")
    try:
        data_metric = data[data['metric_item_label'] == metric].dropna(subset=significant_features + ['value']).copy()
    except KeyError as e:
        print(f"❌ KeyError: {e}")
        continue

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

            if y.nunique() < 2:
                print(f"Skipping Subgroup {feature} = {value} - Only one class present in y.")
                continue

            clf = XGBClassifier(random_state=42)
            clf.fit(X, y)
            score = cross_val_score(clf, X, y, cv=5, scoring='f1').mean()

            if metric not in subgroup_results:
                subgroup_results[metric] = {}
            if feature not in subgroup_results[metric]:
                subgroup_results[metric][feature] = {}

            if feature in category_mappings:
                original_value = [k for k, v in category_mappings[feature].items() if v == value][0]
            else:
                original_value = value

            subgroup_results[metric][feature][original_value] = round(score, 3)

# Display and store results
table = []
print("\nSubgroup Performance Analysis:")
for metric, features in subgroup_results.items():
    table.append([f"Scores of {metric}", "", ""])
    print(f"\nMetric: {metric}")
    for feature, subgroups in features.items():
        print(f"  Feature: {feature}")
        table.append([feature, "", ""])
        for subgroup, score in subgroups.items():
            table.append(["", subgroup, score])
            print(f"    {subgroup}: F1-Score = {score}")

df_table = pd.DataFrame(table, columns=["Metric/Feature", "Subgroup", "F1-Score"])
df_table.to_excel("../Output/7.1 Subgroup_Performance_Analysis_Optimized.xlsx", index=False)
print("\nResults saved to ../Output/7.1 Subgroup_Performance_Analysis_Optimized.xlsx")

# ----------------------- Plotting -----------------------
# Aggregation
aggregated_scores = {}
for metric_data in subgroup_results.values():
    for feature, subgroups in metric_data.items():
        if feature not in aggregated_scores:
            aggregated_scores[feature] = {}
        for subgroup, score in subgroups.items():
            aggregated_scores[feature].setdefault(subgroup, []).append(score)

for feature in aggregated_scores:
    for subgroup in aggregated_scores[feature]:
        aggregated_scores[feature][subgroup] = np.mean(aggregated_scores[feature][subgroup])

# 1️⃣ Heatmap der Subgruppen-F1-Scores nach Metriken
heatmap_entries = []

for metric, feature_dict in subgroup_results.items():
    for feature, subgroups in feature_dict.items():
        for subgroup, score in subgroups.items():
            label = f"{feature} = {subgroup}"
            heatmap_entries.append({
                "Metric": metric,
                "Subgroup": label,
                "F1_Score": score
            })

df_heatmap = pd.DataFrame(heatmap_entries)
heatmap_pivot = df_heatmap.pivot_table(index="Subgroup", columns="Metric", values="F1_Score", aggfunc="mean").fillna(0)

# Plot Heatmap
plt.figure(figsize=(14, max(8, 0.3 * len(heatmap_pivot))))
sns.heatmap(heatmap_pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'F1-Score'})
plt.title("Subgroup F1-Scores by Health Metric")
plt.xlabel("Health Metric")
plt.ylabel("Subgroup")
plt.tight_layout()
plt.savefig("../Output/7.1 Subgroup_vs_Metric_Heatmap.jpg")
plt.show()

# Subplots: Barplots der Subgruppen-F1 nach Feature
# Aggregiere über alle Metriken (wie vorher)
plot_data = []

for metric, feature_dict in subgroup_results.items():
    for feature, subgroups in feature_dict.items():
        for subgroup, score in subgroups.items():
            plot_data.append({
                "Feature": feature,
                "Subgroup": str(subgroup),
                "F1_Score": score
            })

df_bar = pd.DataFrame(plot_data)
avg_df = df_bar.groupby(["Feature", "Subgroup"])["F1_Score"].mean().reset_index()

# Erstelle Subplots je Feature
features = avg_df["Feature"].unique()
n_features = len(features)
n_rows=1
n_cols=6
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows*8), squeeze=False)

for i, feature in enumerate(features):
    row, col = divmod(i, 6)
    ax = axes[row][col]
    data = avg_df[avg_df["Feature"] == feature]
    sns.barplot(data=data, x="Subgroup", y="F1_Score", hue="Subgroup", ax=ax, palette="viridis")
    ax.set_title(f"{feature}")
    ax.set_xlabel("")
    ax.set_ylabel("Avg. F1-Score")
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 1)

    # Leere Achsen ausblenden (falls ungerade Anzahl)
    for j in range(i + 1, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        fig.delaxes(axes[row][col])

plt.suptitle("Average F1-Scores by Subgroup (Grouped by Feature)", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("../Output/7.1 Subgroup_Averages_By_Feature_Subplots.jpg")
plt.show()

