import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


# --------------------------- EINSTELLUNGEN ---------------------------
data = pd.read_csv("../Data/BigCitiesHealth_Cleaned.csv")

features = ['strata_race_label','strata_sex_label','geo_strata_poverty',
            'geo_strata_Segregation','geo_strata_region','geo_strata_PopDensity']
metrics = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths', 
           'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']

models = {
    "XGB": XGBClassifier(random_state=42),
    "RF": RandomForestClassifier(random_state=42),
    "GB": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

# --------------------------- IMPORTANCE SAMMELN ---------------------------
importance_records = []

for metric in metrics:
    print(f"\n📊 Processing Metric: {metric}")
    df_metric = data[data["metric_item_label"] == metric].dropna(subset=features + ['value']).copy()
    if df_metric.empty:
        continue

    for feature in features:
        unique_vals = df_metric[feature].dropna().unique()

        for subgroup_val in unique_vals:
            df_subgroup = df_metric[df_metric[feature] == subgroup_val].copy()
            if df_subgroup.shape[0] < 15:
                continue

            df_encoded = df_subgroup.copy()
            for col in features:
                if df_encoded[col].dtype == object:
                    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

            X = df_encoded[features]
            y = (df_encoded['value'] > df_encoded['value'].median()).astype(int)

            if y.nunique() < 2:
                continue

            for model_name, model in models.items():
                X_proc = X.copy()
                if not hasattr(model, 'feature_importances_') and model_name != "KNN":
                    X_proc = pd.DataFrame(StandardScaler().fit_transform(X_proc), columns=X.columns)

                try:
#################################### Hyperparameter-Tuning #################################
                    if model_name == "XGB":
                        param_grid = {
                            'max_depth': [3, 5],
                            'n_estimators': [50, 100],
                            'learning_rate': [0.05, 0.1]
                        }
                        search = GridSearchCV(
                            estimator=model,
                            param_grid=param_grid,
                            scoring='f1',
                            cv=3,
                            n_jobs=1,
                            error_score='raise'
                        )
                        search.fit(X_proc, y)
                        model = search.best_estimator_

                    elif model_name == "RF":
                        param_grid = {
                            'n_estimators': [50, 100],
                            'max_depth': [None, 5, 10],
                            'max_features': ['sqrt', 'log2']
                        }
                        search = GridSearchCV(
                            estimator=model,
                            param_grid=param_grid,
                            scoring='f1',
                            cv=3,
                            n_jobs=1,
                            error_score='raise'
                        )
                        search.fit(X_proc, y)
                        model = search.best_estimator_

                    elif model_name == "GB":
                        param_grid = {
                            'learning_rate': [0.05, 0.1],
                            'n_estimators': [50, 100],
                            'max_depth': [3, 5]
                        }
                        search = GridSearchCV(
                            estimator=model,
                            param_grid=param_grid,
                            scoring='f1',
                            cv=3,
                            n_jobs=1,
                            error_score='raise'
                        )
                        search.fit(X_proc, y)
                        model = search.best_estimator_

                    elif model_name == "KNN":
                        param_grid = {
                            'n_neighbors': [3, 5, 7],
                            'weights': ['uniform', 'distance']
                        }
                        search = GridSearchCV(
                            estimator=model,
                            param_grid=param_grid,
                            scoring='f1',
                            cv=3,
                            n_jobs=1,
                            error_score='raise'
                        )
                        search.fit(X_proc, y)
                        model = search.best_estimator_

                    # Modelltraining nach Tuning
                    model.fit(X_proc, y)

                    # Feature Importances / Permutation
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                    else:
                        result = permutation_importance(model, X_proc, y, scoring='f1', n_repeats=5, random_state=42)
                        importances = result.importances_mean

                    # Speichern der Ergebnisse...
                    for feat, imp in zip(features, importances):
                        importance_records.append({
                            "Metric": metric,
                            "Feature": feat,
                            "Subgroup": subgroup_val,
                            "Importance": imp
                        })

                except Exception as e:
                    print(f"❌ Error for model {model_name}, metric {metric}, subgroup {subgroup_val}: {e}")
                    continue


# --------------------------- DATAFRAME & PLOTS ---------------------------
# --------------------------- DATAFRAME & PLOTS ---------------------------
importance_df = pd.DataFrame(importance_records)

# Gruppieren und summieren (Subgroup-Wert → Importance aufsummiert über Modelle)
summary_df = (
    importance_df
    .groupby(["Metric", "Feature", "Subgroup"])["Importance"]
    .sum()
    .reset_index()
)

# Plot pro Metrik: Feature auf x-Achse, Importance auf y, Hue = Subgroup
output_dir = "../Output"
os.makedirs(output_dir, exist_ok=True)


for feature in summary_df["Feature"].unique():
    df_feature = summary_df[summary_df["Feature"] == feature]

    plt.figure(figsize=(8, 10))
    sns.barplot(data=df_feature, x="Metric", y="Importance", hue="Subgroup")

    plt.title(f"Subgroup Importance by Metric - {feature}", fontweight="bold")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0,4.5)
    plt.ylabel("Summed Importance")
    plt.tight_layout()
    plt.savefig(f"../Output/7.2 Subgroup Performance - {feature}.png")
    plt.close()

# --------------------------- OPTIONAL: Excel Export ---------------------------
summary_df.to_excel(f"{output_dir}/7.2 Subgroup_Importance_Summary.xlsx", index=False)
print("✅ Fertig: Importance zusammengefasst und geplottet.")
