from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import PartialDependenceDisplay
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Load data
data = pd.read_csv('../../Data/BigCitiesHealth.csv')

# Display available columns
#print("Available columns in the dataset:")
#print(data.columns)

# 2. Define features and target
features = [
    'strata_race_label', 
    'strata_sex_label', 
    'geo_strata_poverty', 
    'geo_strata_region', 
    'geo_strata_PopDensity'
]
targets = [
    'Cardiovascular Disease Deaths',
    'Diabetes Deaths',
    'Injury Deaths',
    'All Cancer Deaths'
]

# Check if target columns exist
missing_targets = [t for t in targets if t not in data['metric_item_label'].unique()]
if missing_targets:
    print(f"\nError: The following target columns are missing: {missing_targets}")
    print("\nAvailable metrics in the dataset:")
    print(data['metric_item_label'].unique())
    exit()

# 3. Prepare data for each metric
df_model = pd.DataFrame()
for target in targets:
    df_metric = data[data['metric_item_label'] == target].dropna(subset=features + ['value']).copy()
    df_metric['target'] = (df_metric['value'] > df_metric['value'].median()).astype(int)
    df_metric['metric_item_label'] = target

    # Encode categorical features using LabelEncoder
    for col in features:
        if df_metric[col].dtype == 'object':
            encoder = LabelEncoder()
            df_metric[col] = encoder.fit_transform(df_metric[col].astype(str))

    df_model = pd.concat([df_model, df_metric], ignore_index=True)


# 4. Encode categorical features
encoders = {}
for col in features:
    if col in df_model.columns:
        df_model[col] = df_model[col].astype(str)  # Convert all to string
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        encoders[col] = le

# 5. Model definition
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)  # Removed use_label_encoder
}

# 6. Evaluate models for each target
results = {}
print(f"\nEvaluating models for targets")
for target in targets:
    df_metric = df_model[df_model['metric_item_label'] == target]
    X = df_metric[features]
    y = df_metric['target']

    if len(y) < 5:
        print(f"Skipping {target} - Not enough samples for cross-validation.")
        continue

    f1 = make_scorer(f1_score, average='binary')
    target_results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring=f1)
        target_results[name] = {'mean_f1': scores.mean(), 'std_f1': scores.std()}

    results[target] = target_results

# 7. Print result comparison
print("\nModel Evaluation (F1-Score):")
for target, res in results.items():
    print(f"\n{target}:")
    for model, metrics in res.items():
        print(f"{model}: F1 = {metrics['mean_f1']:.3f} ± {metrics['std_f1']:.3f}")

# 8. Plot comparing the different Models
model_comparison = []
for target, res in results.items():
    for model, metrics in res.items():
        model_comparison.append([target, model, metrics['mean_f1'], metrics['std_f1']])

comparison_df = pd.DataFrame(model_comparison, columns=['Metric', 'Model', 'Mean F1', 'Std F1'])
plt.figure(figsize=(12, 8))
sns.barplot(data=comparison_df, x='Metric', y='Mean F1', hue='Model')
plt.title('Model Comparison by Mean F1-Score')
plt.savefig("../Outputs/3. Defining best Model - Model Comparison by F1-Scores.jpg")
plt.show()

# 9. Further Checks to Define ML Model best

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring=make_scorer(f1_score))
    accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    precision = cross_val_score(model, X, y, cv=5, scoring='precision').mean()
    recall = cross_val_score(model, X, y, cv=5, scoring='recall').mean()
    roc_auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
    
    return {
        'F1-Score': scores.mean(),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'ROC-AUC': roc_auc
    }

results = {}
for name, model in models.items():
    results[name] = evaluate_model(model, X, y)

print("\nModel Evaluation (Advanced Metrics):")
for model, metrics in results.items():
    print(f"\n{model}:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.3f}")

# Learning Curve
from sklearn.model_selection import learning_curve
import numpy as np

def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='f1', n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, label='Training Score')
    plt.plot(train_sizes, test_mean, label='Validation Score')
    plt.title(f'{title} - Learning Curve')
    plt.xlabel('Training Set Size')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.savefig(f"../Outputs/3. Defining best Model - Learning Curve for {title}.jpg")
    plt.show()

for name, model in models.items():
    plot_learning_curve(model, X, y, name)

# 10. Linearity Check (Correlation Heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(df_model[features + ['target']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature-Target Correlation Heatmap')
plt.savefig("../Outputs/3. Linearity Check - Feature-Target Correlation Heatmap.jpg")
plt.show()

# 10. Linearity Check (Boxplots)
plt.figure(figsize=(15, 8))
sns.boxplot(data=df_model[features])
plt.title('Boxplot of Features')
plt.savefig("../Outputs/3. Linearity Check - Features by Boxplots.jpg")
plt.show()
