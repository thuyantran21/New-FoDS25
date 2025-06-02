
############################################ Anmerung @Thuy ##############################################
# Ich han dini vier codes als eine gstaltet (wie scho mitteilt)
#   das ish halt cool, well denn am afang chash sege, welli metrics / features / farbe / usw. bruchsh
#   und denn direkt alles mit dene "Settings" laufe tuet
# Bi XGB hani 2 Fehler gseh:
#   1.  XGBClassifier mush direkt bruche, ned über e separati funktion, shusht bechunnsh ken plot use
#   2.  Well mer ja jetzt meh metrics hend wie vorher, sind nid all metrics bruchbar
#       -> demit ke Fehlermeldig bechunnsh brucht mer try/exepct
#       -> wenn epis fehlt, denn wird das dur error_score='raise' 
#   3.  du hesh amel n_jobs=-1 verwendet, das hani dur n_jobs=1 ersetzt, well n_jobs=-1 zeme mit de
#       verwendete Packages nid funktioniert.

# Jetzt lauft de Code bi mi ohni Fehler und all Plots werdet erstellt. 
# Ich ha dir hoffentlich chönne helfe :)
# Aber wenn du irgendepis andersh mache wottsh ish das natürlich überhaupt kes ding!


############################################ Needed Imports ############################################
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import learning_curve, StratifiedKFold
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin

############################################ General Settings ###########################################
# Load Data Set
data = pd.read_csv("../Data/BigCitiesHealth.csv")

# Prepare dataset
targets = ['Low Birthweight', 'High Blood Pressure', 'Life Expectancy', 'Adult Mental Distress', 
           'Infant Deaths', 'Lung Cancer Deaths', 'Maternal Deaths']
features = ['strata_race_label','strata_sex_label','geo_strata_poverty','geo_strata_Segregation',
            'geo_strata_region','geo_strata_PopDensity','geo_strata_Population']

colors = itertools.cycle(['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'cyan'])


########################################### Logistic Regression ########################################
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
logistic_df.to_excel("../Output/3.2 ML_Model_Results.xlsx", index=False)

#Learning curve for all targets
plt.figure(figsize=(10, 6))

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

    try:                                                                            # Notwendig, da neue metrics nicht alle genug samples haben für verlgeich
        train_sizes, train_scores, test_scores = learning_curve(
            pipeline,
            X_lc,
            y_lc,
            train_sizes=np.linspace(0.1, 1.0, 5),
            cv=StratifiedKFold(n_splits=5),
            scoring='f1',
            n_jobs=1,
            error_score='raise'  # Raise instead of NaN, so we can catch it below
        )
    except ValueError as e:
        #print(f"Skipping {target} due to learning_curve error: {e}")
        continue

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # Add both lines to plot (training + validation)
    color = next(colors)
    plt.plot(train_sizes, train_mean, linestyle='--', marker='o', color=color, label=f"{target} - Train")
    plt.plot(train_sizes, test_mean, linestyle='-', marker='s', color=color, label=f"{target} - Val")

# Finalize and save plot
plt.title("Learning Curve - Logistic Regression (All Targets)")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../Output/3.2 Learning Curve Logistic Regression.png")
plt.show()


############################################### KNN ####################################################
# Label encode categorical features
for col in features:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# Store results
knn_results = []

# Fit model for each target
print ("\nKNN Results:")
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

knn_df.to_excel("../Output/3.2 KNN_Results.xlsx", index=False)


# Learning Curve
plt.figure(figsize=(10, 6))

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
        scoring='f1', n_jobs=1
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
plt.savefig("../Output/3.2 Learning Curve KNN.png")
plt.show()


############################################### Random Forest ##########################################
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
results_df.to_excel("../Output/3.2 ML_Model_Results.xlsx", index=False)

#Learning curve
plt.figure(figsize=(10, 6))

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
        scoring='f1', n_jobs=1, error_score='raise'
    )
    color=next(colors)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), linestyle='--', marker='o', color=color, label=f"{target} - Train")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), linestyle='-', marker='s', color=color, label=f"{target} - Val")

plt.title("Learning Curve - Random Forest (All Targets)")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../Output/3.2 Learning Curve RF.png")
plt.show()



############################################## XGBoost #################################################
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
# Helping function
def has_valid_stratification(y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits)
    for _, val_idx in skf.split(np.zeros(len(y)), y):
        y_fold = y[val_idx] if isinstance(y, np.ndarray) else y.iloc[val_idx]
        if len(np.unique(y_fold)) < 2:
            return False
    return True

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
xgb_manual_df.to_excel("../Output/3.2 ML_Model_Results.xlsx", index=False)

#Learning curve
plt.figure(figsize=(10, 6))

# Loop through each target and plot curves
for i, target in enumerate(targets):
    df = data[data["metric_item_label"] == target].dropna(subset=features + ['value']).copy()
    df['target'] = (df['value'] > df['value'].median()).astype(int)

    X = df[features]
    y = df['target']

    model = XGBClassifier(                      # Hier direkt Classifier verwenden und dafür oberer Abschnitt zu ...(XGBClassifier, ...) entfernen
        eval_metric='logloss',
        random_state=42
    )
    if y.nunique() < 2:
        print(f"Skipping {target} – only one class ({y.unique()[0]}) in full target data")
        continue

    try:                                                                    # Notwendig, da nicht alle Metrics brauchbar
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y,
            train_sizes=np.linspace(0.1, 1.0, 5),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=1
        )
    except ValueError as e:
        print(f"⚠️ Skipping {target} due to CV error: {e}")
        continue

    # Plot training + validation curves
    color = next(colors)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), linestyle='--', marker='o',
             color=color, label=f"{target} - Train")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), linestyle='-', marker='s',
             color=color, label=f"{target} - Val")

#Finalize and save
plt.title("Learning Curve - XGBoost (All Targets)")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../Output/3.2 Learning Curve XGBoost.png")
plt.show()
