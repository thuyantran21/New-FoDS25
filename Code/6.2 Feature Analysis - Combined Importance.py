import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import os
from sklearn.metrics import r2_score

# --------------------------- EINSTELLUNGEN ---------------------------
data = pd.read_csv("../Data/BigCitiesHealth_Cleaned.csv")

features = ['strata_race_label','strata_sex_label','geo_strata_poverty','geo_strata_Segregation','geo_strata_region','geo_strata_PopDensity']
metrics = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths', 'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']

models = {
    "XGBoost": XGBClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

param_grids = {
    "XGBoost": {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.1, 0.3]},
    "Random Forest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
    "Gradient Boosting": {"n_estimators": [50, 100], "learning_rate": [0.1, 0.3], "max_depth": [3, 5]},
    "KNN": {"n_neighbors": [3, 5, 7]}
}

# --------------------------- INITIALISIERUNG ---------------------------
overall_importance = {f: 0 for f in features}
model_importance_store = {}
ml_results = []
combined_all_metric_importances = []

# --------------------------- MODELLSCHLEIFE ---------------------------
for model_name, model in models.items():
    print(f"\n🔍 Modell: {model_name}")
    combined_importance = {f: 0 for f in features}

    for metric in metrics:
        print(f"\n📊 Processing Metric: {metric}")
        df = data[data['metric_item_label'] == metric].dropna(subset=features + ['value']).copy()
        if df.empty:
            continue

        for col in features:
            df[col] = LabelEncoder().fit_transform(df[col])

        X = df[features]
        y = (df['value'] > df['value'].median()).astype(int)

        if model_name == "KNN":
            X = pd.DataFrame(StandardScaler().fit_transform(X), columns=features)

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

        grid = GridSearchCV(model, param_grids[model_name], scoring='f1', cv=3, n_jobs=1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        ml_results.append({
            "Model": model_name,
            "Metric": metric,
            "F1-Score": round(f1, 3),
            "R2": round(r2, 3)
            })

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            result = permutation_importance(model, X, y, scoring='f1', n_repeats=10, random_state=42)
            importances = result.importances_mean

        for f, val in zip(features, importances):
            combined_importance[f] += val
            overall_importance[f] += val
            combined_all_metric_importances.append({
                "Model": model_name,
                "Metric": metric,
                "Feature": f,
                "Importance": val
            })

    sorted_combined = dict(sorted(combined_importance.items(), key=lambda x: x[1], reverse=True))
    plt.figure(figsize=(8, 6))
    plt.bar(sorted_combined.keys(), sorted_combined.values())
    plt.title(f"{model_name} - Combined Feature Importance (all metrics)")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Total Importance Score")
    plt.tight_layout()
    plt.savefig(f"../Output/6.2 Combined Importance {model_name}.pdf")

    model_importance_store[model_name] = pd.Series(combined_importance)

# --------------------------- GESAMTPLOT ---------------------------
sorted_overall = dict(sorted(overall_importance.items(), key=lambda x: x[1], reverse=True))
plt.figure(figsize=(9, 6))
plt.bar(sorted_overall.keys(), sorted_overall.values())
plt.title("Overall Feature Importance (Aggregated across all models & metrics)", fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Summed Importance Score")
plt.tight_layout()
plt.savefig("../Output/6.2 Overall Feature Importance.jpg")

# --------------------------- METRIK-WEISE PLOTS ---------------------------
print("\n📊 Generating combined Plots using all models for each Metric...")

for metric in metrics:
    df_metric_imp = pd.DataFrame([
        row for row in combined_all_metric_importances if row["Metric"] == metric
    ])

    if df_metric_imp.empty:
        continue

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_metric_imp, x="Feature", y="Importance", hue="Model")
    plt.title(f"Feature Importance für Metrik: {metric}", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"../Output/6.2 Feature Importance {metric}.jpg")
# --------------------------- EXCEL-EXPORT (KORRIGIERT NACH DEINER LOGIK) ---------------------------
print("\n📁 Erstelle finale Excel-Datei mit allen Ergebnissen...")

# 1️⃣ Sheet: Lange Tabelle – alles, nichts aggregiert
sheet1_df = pd.DataFrame(combined_all_metric_importances)

# 2️⃣ Sheet: Pivot – pro Metrik und Feature, Modelle als Spalten
pivot_df = (
    sheet1_df
    .pivot_table(index=["Metric", "Feature"], columns="Model", values=["Importance"] , aggfunc="sum")
    .fillna(0)
    .round(4)
)

# 3️⃣ Sheet: Vollständig aggregiert – über alle Modelle & Metriken
aggregated_df = (
    sheet1_df
    .groupby("Feature")["Importance"]
    .sum()
    .reset_index()
    .rename(columns={"Importance": "Importance (all health metrics)"})
    .sort_values(by="Importance (all health metrics)", ascending=False)
)

# Zielpfad
output_dir = "../Output"
os.makedirs(output_dir, exist_ok=True)
final_excel_path = os.path.join(output_dir, "6.2_Model_Overview_Complete.xlsx")

# Exportieren mit korrekten Sheetnamen
with pd.ExcelWriter(final_excel_path, engine="openpyxl") as writer:
    sheet1_df.to_excel(writer, sheet_name="Feature Importance", index=False)
    pivot_df.to_excel(writer, sheet_name="Combined Feature Importance")
    aggregated_df.to_excel(writer, sheet_name="Aggregated Feature Importance", index=False)

print(f"✅ Excel-Datei erfolgreich erstellt: {final_excel_path}")
