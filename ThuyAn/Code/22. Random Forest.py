import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load dataset
data = pd.read_csv("../../Data/BigCitiesHealth.csv")

# Define target outcomes and features
targets = ['Cardiovascular Disease Deaths', 'Diabetes Deaths', 'Injury Deaths', 'All Cancer Deaths']
features = [
    'strata_race_label',
    'strata_sex_label',
    'geo_strata_poverty',
    'geo_strata_Segregation',
    'geo_strata_region',
    'geo_strata_PopDensity',
    'geo_strata_Population'
]

# Encode categorical features numerically
for col in features:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# Store model results
rf_results = []

for target in targets:
    # Filter data for the current target (no dropna)
    df = data[data["metric_item_label"] == target].copy()

    # Remove rows with missing target values only
    df = df[df['value'].notna()]
    df['target'] = (df['value'] > df['value'].median()).astype(int)

    # Split features and labels
    X = df[features]
    y = df['target']

    # Drop rows where features contain missing values (to avoid model crash)
    valid_rows = X.dropna().index
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Random Forest pipeline
    pipeline = Pipeline([
        ("model", RandomForestClassifier(random_state=42))
    ])

    # Hyperparameter grid (simplified)
    param_grid = {
        'model__n_estimators': [100],
        'model__max_depth': [None, 10],
        'model__min_samples_split': [2],
        'model__max_features': ['sqrt']
    }

    # Grid search
    grid = GridSearchCV(pipeline, param_grid, scoring='f1', cv=5)
    grid.fit(X_train, y_train)

    # Predictions
    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    rf_results.append({
        'Target': target,
        #'Best Params': grid.best_params_,
        'F1 Score': round(report['1']['f1-score'], 3),
        'Precision': round(report['1']['precision'], 3),
        'Recall': round(report['1']['recall'], 3),
        'Accuracy': round(accuracy, 3),
        'ROC-AUC': round(roc_auc, 3)
    })

# Compile results
results_df = pd.DataFrame(rf_results)

# Output results
print("\nRandom Forest Results (without dropna):")
#print(results_df)
print(results_df.to_string(index=False))

print("\nAverage Results:")
print(results_df[['F1 Score', 'Precision', 'Recall', 'Accuracy', 'ROC-AUC']].mean())

# Save results to Excel
results_df.to_excel("../Outputs/ML_Model_Results.xlsx", index=False)

#Learning curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'orange', 'red']

for i, target in enumerate(targets):
    df_rf = data[data["metric_item_label"] == target].dropna(subset=features + ['value']).copy()
    df_rf['target'] = (df_rf['value'] > df_rf['value'].median()).astype(int)

    X = df_rf[features]
    y = df_rf['target']

    model = RandomForestClassifier(random_state=42)

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=StratifiedKFold(n_splits=5),
        scoring='f1', n_jobs=-1
    )

    plt.plot(train_sizes, np.mean(train_scores, axis=1), linestyle='--', marker='o', color=colors[i], label=f"{target} - Train")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), linestyle='-', marker='s', color=colors[i], label=f"{target} - Val")

plt.title("Learning Curve - Random Forest (All Targets)")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../Outputs/Learning_Curve_RF.png")
plt.show()
