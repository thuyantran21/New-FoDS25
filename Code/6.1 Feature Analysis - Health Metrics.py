import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# ---------------- SETTINGS ----------------
models = {
    "XGBoost": XGBRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "KNN": KNeighborsRegressor()
}

param_grids = {
    "XGBoost": {"n_estimators": [50], "max_depth": [3]},
    "Random Forest": {"n_estimators": [100], "max_depth": [None]},
    "Gradient Boosting": {"n_estimators": [100], "learning_rate": [0.1]},
    "KNN": {"n_neighbors": [5]}
}

features = ['strata_race_label','strata_sex_label','geo_strata_poverty','geo_strata_Segregation','geo_strata_region','geo_strata_PopDensity']
metrics = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths', 'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']

data = pd.read_csv("../Data/BigCitiesHealth_Cleaned.csv")
os.makedirs("../Output", exist_ok=True)

# Ergebnisliste für Excel
importance_records = []

# ---------------- LOOP PER MODEL ----------------
for model_name, base_model in models.items():
    print(f"\n🔍 Running Model: {model_name}")
    
    # Plot Setup: 1 Zeile pro Metrik, 3 Spalten
    fig, axes = plt.subplots(len(metrics), 3, figsize=(18, len(metrics) * 4))
    combined_importance = {feature: 0 for feature in features}
    
    for i, metric in enumerate(metrics):
        metric_data = data[data['metric_item_label'] == metric].dropna(subset=features + ['value']).copy()

        # Label-Encoding
        for col in features:
            metric_data[col] = LabelEncoder().fit_transform(metric_data[col])

        X = metric_data[features]
        y = metric_data['value']

        if model_name == "KNN":
            X = pd.DataFrame(StandardScaler().fit_transform(X), columns=features)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # GridSearch (n_jobs=1 zur Vermeidung von Loky-Warnungen)
        grid = GridSearchCV(base_model, param_grids.get(model_name, {}), cv=3, scoring='r2', n_jobs=1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        # Permutation Importance
        result = permutation_importance(best_model, X, y, n_repeats=10, random_state=42, scoring='r2')
        importances = result.importances_mean

        # Accumulate for combined view
        for feat, val in zip(features, importances):
            combined_importance[feat] += val
            importance_records.append({
                "Model": model_name,
                "Metric": metric,
                "Feature": feat,
                "Importance": round(val, 5),
                "R2_Score": round(r2, 3)
            })

        # Plot 1: Feature Importance
        ax0 = axes[i, 0]
        ax0.bar(features, importances)
        ax0.set_title(f"{metric} - Feature Importance")
        ax0.tick_params(axis='x', rotation=45)

        # Plot 2: Feature Distribution
        ax1 = axes[i, 1]
        for feat in features:
            sns.histplot(metric_data[feat], kde=True, label=feat, ax=ax1, alpha=0.5)
        ax1.set_title(f"{metric} - Feature Distribution")
        ax1.legend()

        # Plot 3: Model Performance
        ax2 = axes[i, 2]
        ax2.scatter(y_test, y_pred, alpha=0.6)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax2.set_xlabel("True Values")
        ax2.set_ylabel("Predictions")
        ax2.set_title(f"{metric} - R² = {r2:.2f}")

    plt.tight_layout(h_pad=1.5, w_pad=2)
    plt.savefig(f"../Output/6.1_Model_Overview_{model_name}.pdf")
    plt.savefig(f"../Output/6.1_Model_Overview_{model_name}.jpg")
    plt.show()

# ---------------- EXPORT TO EXCEL ----------------
importance_df = pd.DataFrame(importance_records)
excel_path = "../Output/6.1_Model_Importances.xlsx"
importance_df.to_excel(excel_path, index=False)
print(f"\n✅ Ergebnisse erfolgreich exportiert nach: {excel_path}")
