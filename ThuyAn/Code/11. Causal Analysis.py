import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the cleaned dataset
data = pd.read_csv("../../Data/BigCitiesHealth_Cleaned.csv")

# Define target metrics (outcome variables)
metrics = [
    'Cardiovascular Disease Deaths',
    'Diabetes Deaths',
    'Injury Deaths',
    'All Cancer Deaths'
]

# Define two feature sets
significant_features = [
    'strata_race_label',
    'strata_sex_label',
    'geo_strata_poverty',
    'geo_strata_PopDensity',
    'geo_strata_region'
]

all_features = significant_features + ['geo_strata_Segregation', 'geo_strata_Population']

# Store results
causal_results_significant = []
causal_results_all = []

# Function to perform Cross-Correlation Analysis
def cross_correlation_analysis(data, features, metrics, result_storage):
    for metric in metrics:
        print(f"\nAnalyzing Cross-Correlation for Metric: {metric}")
        data_metric = data[data['metric_item_label'] == metric].dropna(subset=features + ['value']).copy()

        for feature in features:
            print(f"\n\nTesting Cross-Correlation: {feature} -> {metric}")

            ts_data = data_metric[[feature, 'value']].dropna()
            if ts_data[feature].dtype == 'object':
                ts_data[feature] = pd.factorize(ts_data[feature])[0]

            cross_corr_values = [ts_data[feature].corr(ts_data['value'].shift(lag)) for lag in range(1, 5)]
            max_corr = max(cross_corr_values, key=abs)
            best_lag = cross_corr_values.index(max_corr) + 1

            result_storage.append({
                'Metric': metric,
                'Feature': feature,
                'Max Cross-Correlation': max_corr,
                'Optimal Lag': best_lag,
                'Causal': 'Yes' if abs(max_corr) > 0.05 else 'No'
            })

# Run analysis for both sets
cross_correlation_analysis(data, significant_features, metrics, causal_results_significant)
cross_correlation_analysis(data, all_features, metrics, causal_results_all)

# Save results to Excel
results_significant = pd.DataFrame(causal_results_significant)
results_all = pd.DataFrame(causal_results_all)

with pd.ExcelWriter("../Outputs/9. Causal_Impact_Analysis_Comparison.xlsx") as writer:
    results_significant.to_excel(writer, sheet_name="Significant_Features", index=False)
    results_all.to_excel(writer, sheet_name="All_Features", index=False)

# Visualize the results
plt.figure(figsize=(12, 6))
sns.countplot(data=results_significant, x='Feature', hue='Causal')
plt.title("Cross-Correlation: Significant Features (Causal Analysis)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../Outputs/9. Causal_Impact_Significant_Results.png")

plt.figure(figsize=(12, 6))
sns.countplot(data=results_all, x='Feature', hue='Causal')
plt.title("Cross-Correlation: All Features (Causal Analysis)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../Outputs/9. Causal_Impact_All_Results.png")
plt.show()

print("Causal Impact Analysis using Cross-Correlation completed. Results saved to Causal_Impact_Analysis_Comparison.xlsx.")
