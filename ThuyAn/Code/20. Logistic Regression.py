import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Load data
data = pd.read_csv("../../Data/BigCitiesHealth.csv")

# Updated targets (corrected spelling)
targets = [
    'Infant Deaths', 
    'Life Expectancy',
    'Low Birthweight',
    'Adult Mental Distress',
    'High Blood Pressure',
    'Lung Cancer Deaths',
    'Maternal Deaths'
]

features = [
    'strata_race_label',
    'strata_sex_label',
    'geo_strata_poverty',
    'geo_strata_Segregation',
    'geo_strata_region',
    'geo_strata_PopDensity',
    'geo_strata_Population'
]

# Label encode categorical features
for col in features:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# Store results
knn_results = []

# Fit model for each target
for target in targets:
    df = data[data["metric_item_label"] == target].dropna(subset=features + ['value']).copy()
    df['target'] = (df['value'] > df['value'].median()).astype(int)

    X = df[features]
    y = df['target']

    if y.nunique() < 2:
        print(f" Skipping {target} – only one class present.")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ])

    param_grid = {
        'model__n_neighbors': [3, 5, 7, 9],
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['euclidean']
    }

    grid = GridSearchCV(pipeline, param_grid, scoring='f1', cv=5)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    knn_results.append({
    'Target': target,
    'Best Params': str(grid.best_params_),
    'F1 Score': round(report['1']['f1-score'], 3),
    'Precision': round(report['1']['precision'], 3),
    'Recall': round(report['1']['recall'], 3),
    'Accuracy': round(accuracy, 3),
    'ROC-AUC': round(roc_auc, 3)
})

# Show & save results
knn_df = pd.DataFrame(knn_results)
print(knn_df)

print("\nAverage Results:")
print(knn_df[['F1 Score', 'Precision', 'Recall', 'Accuracy', 'ROC-AUC']].mean())

knn_df.to_excel("../Outputs/KNN_Results.xlsx", index=False)


# Learning Curve
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
colors = itertools.cycle(['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'cyan'])

for target in targets:
    df_lc = data[data["metric_item_label"] == target].dropna(subset=features + ['value']).copy()
    df_lc['target'] = (df_lc['value'] > df_lc['value'].median()).astype(int)

    X = df_lc[features]
    y = df_lc['target']

    if y.nunique() < 2:
        continue

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ])

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=StratifiedKFold(n_splits=5),
        scoring='f1', n_jobs=-1
    )

    c = next(colors)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), linestyle='--', marker='o', color=c, label=f"{target} - Train")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), linestyle='-', marker='s', color=c, label=f"{target} - Val")

plt.title("Learning Curve - KNN (All Targets)")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../Outputs/Learning_Curve_KNN.png")
plt.show()
