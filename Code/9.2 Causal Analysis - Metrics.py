import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load the cleaned dataset
data = pd.read_csv("../Data/BigCitiesHealth_Cleaned.csv")

# Define target metrics (outcome variables)

metrics = ['Cardiovascular Disease Deaths','Diabetes Deaths','Injury Deaths','All Cancer Deaths']

#metrics = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths', 'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']

# Define feature set
features = [
    'strata_race_label',
    'strata_sex_label',
    'geo_strata_poverty',
    'geo_strata_Segregation',
    'geo_strata_region',
    'geo_strata_PopDensity'
]

# Store results
causal_results_all = {}

# Function to perform Lagged Cross-Correlation Analysis (Kausalität)
def lagged_cross_correlation(data, features, metric, max_lag=5, threshold=0.05):
    causal_results = []

    print(f"\nAnalyzing Causal Impact for Metric: {metric}")
    data_metric = data[data['metric_item_label'] == metric].dropna(subset=features + ['value']).copy()

    for feature in features:
        print(f"\nTesting Causal Impact: {feature} -> {metric}")
        ts_data = data_metric[[feature, 'value']].dropna()

        # Factorize categorical features
        if ts_data[feature].dtype == 'object':
            ts_data[feature] = pd.factorize(ts_data[feature])[0]

        # Calculate Cross-Correlation for multiple lags
        cross_corr_values = []
        for lag in range(1, max_lag + 1):
            if len(ts_data) > lag:
                shifted_values = ts_data['value'].shift(lag).dropna()
                feature_values = ts_data[feature][:len(shifted_values)]
                correlation = np.corrcoef(feature_values, shifted_values)[0, 1]
                cross_corr_values.append((lag, correlation))
            else:
                cross_corr_values.append((lag, 0))

        # Find the highest cross-correlation (absolute value)
        max_lag, max_corr = max(cross_corr_values, key=lambda x: abs(x[1]))

        # Determine if causal (based on threshold)
        causal = 'Yes' if abs(max_corr) > threshold else 'No'

        causal_results.append({
            'Feature': feature,
            'Max Correlation': round(max_corr, 4),
            'Optimal Lag': max_lag,
            'Causal': causal
        })

    return causal_results

# Perform analysis for each metric
for metric in metrics:
    causal_results = lagged_cross_correlation(data, features, metric)
    causal_results_all[metric] = pd.DataFrame(causal_results)

# Save all results to Excel (separate sheets for each metric)
output_path = "../Output/9.2 Causal_Impact-Lagged_Correlation_Per_Metric.xlsx"
with pd.ExcelWriter(output_path) as writer:
    for metric, df in causal_results_all.items():
        df.to_excel(writer, sheet_name=metric, index=False)

print(f"\nLagged Cross-Correlation Analysis completed. Results saved to {output_path}.")

# Visualisierung der kausalen Features (pro Metric)
fig, axes = plt.subplots(len(metrics), 1, figsize=(12, len(metrics) * 4), constrained_layout=True)
fig.suptitle("Lagged Cross-Correlation Analysis (Causal Features by Metric)", fontsize=16)

for ax, metric in zip(axes, metrics):
    df_metric = causal_results_all[metric]
    sns.countplot(data=df_metric, x='Feature', hue='Causal', palette='viridis', ax=ax)
    ax.set_title(f"Causal Impact for {metric}")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)

#plt.tight_layout()
plt.savefig("../Output/9.2 Causal Impact - Lagged Correlation Per Metric.png")
plt.show()
