import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import PartialDependenceDisplay
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, make_scorer, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy.stats import f_oneway, pearsonr, spearmanr



#########################################################################################################################
################################################## General Settings #####################################################
#########################################################################################################################

# Load dataset (cleaned)
data = pd.read_csv("../Data/BigCitiesHealth_Cleaned.csv")

# Settings
features = ['strata_race_label', 'strata_sex_label', 'geo_strata_poverty', 'geo_strata_Segregation', 'geo_strata_region', 'geo_strata_PopDensity']
metrics = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths', 'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']


#########################################################################################################################
############################################# 6. Evaluation of Metrics ##################################################
#########################################################################################################################

relevant_metrics = metrics

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
#plt.savefig("../Outputs/6. Heatmap Correlation.jpg")
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
#plt.savefig("../Outputs/6. Metric Analysis - Linear Regression.jpg")
plt.show()


# Summiere die Importances
combined_importances = rf_importances.add(xgb_importances, fill_value=0).sort_values(ascending=False)
print(combined_importances)
# Feature Importance Plot
plt.figure(figsize=(18, 12))
combined_importances.sort_values().plot(kind='barh')
plt.title('Combined Feature Importance (RandomForest + XGBoost)')
#plt.savefig("../Outputs/6. Overall Metric Analysis.jpg")
plt.show()


# Überarbeiteter Teil: Signifikanzanalyse
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



#########################################################################################################################
################################################    7. ML Modell    #####################################################
#########################################################################################################################
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
    #axes[i, 1].legend()
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

plt.tight_layout(pad=1.0)   
#plt.savefig("../Outputs/7.1 ML Modell Results Overview.pdf", format='pdf')
#plt.savefig("../Outputs/7.1 ML Modell Results Overview.jpg", format='jpg')

plt.show()

# Calculate average importance across all metrics
average_importance = {feature: value / len(metrics) for feature, value in combined_importance.items()}

# Plotting Aggregated Feature Importance
plt.figure(figsize=(10, 6))
plt.bar(average_importance.keys(), average_importance.values())
plt.title("Aggregated Feature Importance over all metrics:")
plt.xlabel("Features")
plt.ylabel("Average Importance")
plt.tight_layout(pad=2.0)
#plt.savefig("../Outputs/7.1 Aggregated_Feature_Importance.pdf", format='pdf')
plt.show()
print("\nAggregated Feature Importance over all metrics:")
for feature, value in average_importance.items():
    print(f"{feature}: {value:.4f}")

# Save Results to Excel
results_df_classification = pd.DataFrame(model_results_classification).T
results_df_regression = pd.DataFrame(model_results_regression).T
importance_df = pd.DataFrame(feature_importances)

average_importance_df = pd.DataFrame.from_dict(average_importance, orient='index', columns=['Average Importance'])

with pd.ExcelWriter("../Outputs/7.1 ML_Results_Optimized.xlsx") as writer:
    results_df_classification.to_excel(writer, sheet_name="Classification_Results")
    results_df_regression.to_excel(writer, sheet_name="Regression_Results")
    importance_df.to_excel(writer, sheet_name="Feature_Importance")
    average_importance_df.to_excel(writer, sheet_name="Aggregated_Importance")

print("Results saved to ML_Results_Optimized.xlsx")
