from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Load data
data = pd.read_csv("../Data/BigCitiesHealth.csv")

# Targets and features
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

# Label encode all categorical features
for col in features:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# Store updated logistic regression results
logistic_results = []

# Loop over targets
for target in targets:
    # Drop rows with missing values
    df = data[data["metric_item_label"] == target].dropna(subset=features + ['value']).copy()
    df['target'] = (df['value'] > df['value'].median()).astype(int)

    X = df[features]
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Logistic regression pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    # Hyperparameter grid
    param_grid = {
        'model__C': [0.01, 0.1, 1, 10],
        'model__penalty': ['l2'],
        'model__solver': ['liblinear', 'saga']
    }

    # Run GridSearchCV
    grid = GridSearchCV(pipeline, param_grid, scoring='f1', cv=5)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]  # Needed for ROC-AUC

    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    logistic_results.append({
        'Target': target,
        #'Best Params': grid.best_params_,
        'F1 Score': round(report['1']['f1-score'], 3),
        'Precision': round(report['1']['precision'], 3),
        'Recall': round(report['1']['recall'], 3),
        'Accuracy': round(accuracy, 3),
        'ROC-AUC': round(roc_auc, 3)
    })

# Display results
logistic_df = pd.DataFrame(logistic_results)
print(logistic_df)

#Get average resulats for F1, Precision, Recall, Accuracy and ROC-AUC
print("\nAverage Results:")
logistic_avg = logistic_df[['F1 Score', 'Precision', 'Recall', 'Accuracy', 'ROC-AUC']].mean()
logistic_avg = pd.DataFrame(logistic_avg).T
logistic_avg['Target'] = 'Average'
print(logistic_avg)

# Save results to Excel
logistic_df.to_excel("../Outputs/ML_Model_Results.xlsx", index=False)

#Learning curve
from sklearn.model_selection import learning_curve, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler

# Define the model
#model = LogisticRegression(max_iter=1000, class_weight="balanced")
# Define the scoring function
#scoring = make_scorer(f1_score)
# Define the learning curve
#train_sizes, train_scores, test_scores = learning_curve(
 #   model,
  #  X,
   # y,
   # train_sizes=np.linspace(0.1, 1.0, 10),
   # cv=5,
   # scoring=scoring
#)
# Calculate the mean and standard deviation of the training and test scores
#train_scores_mean = np.mean(train_scores, axis=1)
#train_scores_std = np.std(train_scores, axis=1)
#test_scores_mean = np.mean(test_scores, axis=1)
#test_scores_std = np.std(test_scores, axis=1)

# Plot the learning curve
#plt.figure(figsize=(10, 8))
#plt.plot(train_sizes, train_scores_mean, label="Training F1 Score", color="blue")
#plt.plot(train_sizes, test_scores_mean, label="Validation F1 Score", color="red")
#plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="blue", alpha=0.2)
#plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color="orange", alpha=0.2)
#plt.title("Learning Curve")
#plt.xlabel("Training Size")
#plt.ylabel("F1 Score")
#plt.legend()
#plt.grid()
#plt.show()

#Learning curve for all targets
plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'orange', 'red']

# Loop over all 4 targets
for i, target in enumerate(targets):
    df_lc = data[data["metric_item_label"] == target].dropna(subset=features + ['value']).copy()
    df_lc['target'] = (df_lc['value'] > df_lc['value'].median()).astype(int)

    X_lc = df_lc[features]
    y_lc = df_lc['target']

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    train_sizes, train_scores, test_scores = learning_curve(
        pipeline,
        X_lc,
        y_lc,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=StratifiedKFold(n_splits=5),
        scoring='f1',
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # Add both lines to plot (training + validation)
    plt.plot(train_sizes, train_mean, linestyle='--', marker='o', color=colors[i], label=f"{target} - Train")
    plt.plot(train_sizes, test_mean, linestyle='-', marker='s', color=colors[i], label=f"{target} - Val")

# Finalize and save plot
plt.title("Learning Curve - Logistic Regression (All Targets)")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../Outputs/Learning_Curve_Logistic_Regression.png")
plt.show()
