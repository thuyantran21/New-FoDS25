# Re-import necessary packages for manual XGBoost tuning
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

data = pd.read_csv("../Data/BigCitiesHealth.csv")

# Prepare dataset
targets = ['Low Birthweight', 'Diabetes Deaths', 'Maternal Deaths', 'All Cancer Deaths','Life Expectancy', 'Infant Deaths',]
features = [
    'strata_race_label',
    'strata_sex_label',
    'geo_strata_poverty',
    'geo_strata_Segregation',
    'geo_strata_region',
    'geo_strata_PopDensity',
    'geo_strata_Population'
]

# Label encode features
for col in features:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# Manual hyperparameter grid
param_grid = [
    {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.3},
    {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.3},
    {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.5},
    {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.5},
]

# Store results
xgb_manual_results = []

for target in targets:
    df = data[data["metric_item_label"] == target].copy()
    df = df[df["value"].notna()]
    df["target"] = (df["value"] > df["value"].median()).astype(int)

    X = df[features]
    y = df["target"]

    valid_rows = X.dropna().index
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    best_f1 = 0
    best_result = {}

    for params in param_grid:
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            **params
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = report['1']['f1-score']
        precision = report['1']['precision']
        recall = report['1']['recall']
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        if f1 > best_f1:
            best_f1 = f1
            best_result = {
                'Target': target,
                #'Best Params': params,
                'F1 Score': round(f1, 3),
                'Precision': round(precision, 3),
                'Recall': round(recall, 3),
                'Accuracy': round(accuracy, 3),
                'ROC-AUC': round(roc_auc, 3)
            }

    xgb_manual_results.append(best_result)



# Display results
xgb_manual_df = pd.DataFrame(xgb_manual_results)
print(xgb_manual_df)
xgb_manual_df

# Average of results
print("\nAverage Results:")
print(xgb_manual_df[['F1 Score', 'Precision', 'Recall', 'Accuracy', 'ROC-AUC']].mean())

# Save results to Excel
xgb_manual_df.to_excel("../Outputs/ML_Model_Results.xlsx", index=False)

#Learning curve
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import make_scorer, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

class SklearnCompatibleXGB(XGBClassifier, BaseEstimator, ClassifierMixin):
    pass

plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'orange', 'red']

# Loop through each target and plot curves
for i, target in enumerate(targets):
    df = data[data["metric_item_label"] == target].dropna(subset=features + ['value']).copy()
    df['target'] = (df['value'] > df['value'].median()).astype(int)

    X = df[features]
    y = df['target']

    model = SklearnCompatibleXGB(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=StratifiedKFold(n_splits=5),
        scoring='f1',
        n_jobs=-1
    )

    # Plot training + validation curves
    plt.plot(train_sizes, np.mean(train_scores, axis=1), linestyle='--', marker='o',
             color=colors[i], label=f"{target} - Train")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), linestyle='-', marker='s',
             color=colors[i], label=f"{target} - Val")

#Finalize and save
plt.title("Learning Curve - XGBoost (All Targets)")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../Outputs/Learning_Curve_XGBoost.png")
plt.show()