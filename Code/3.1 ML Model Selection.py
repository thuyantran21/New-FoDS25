import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve
from sklearn.metrics import f1_score, make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skorch import NeuralNetClassifier
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# Load data
data = pd.read_csv('../Data/BigCitiesHealth.csv')
features = ['strata_race_label', 'strata_sex_label', 'geo_strata_poverty', 'geo_strata_Segregation', 'geo_strata_region', 'geo_strata_PopDensity']
targets = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths', 'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']


# Encode categorical features
for col in features:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# Define NKN model
class NKN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(NKN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.kernel = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.sin(self.kernel(x))
        return self.out(x)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Ridge Classifier': RidgeClassifier(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),                                   #'Decision Tree is pretty similar to Random Forest and therefor was not included, 'Linear Regression' shows no result as not predictable,
    'Naive Bayes': GaussianNB(),
    'SVM (Linear)': SVC(kernel='linear', probability=True),
    'SVM (RBF)': SVC(kernel='rbf', probability=True),
    'KNN': KNeighborsClassifier(),
    'MLP': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=300, random_state=42),
    'NKN': NeuralNetClassifier(NKN, module__input_dim=len(features), max_epochs=20, lr=0.01,
                               optimizer=torch.optim.Adam, verbose=0)
}

# Model evaluation
model_scores = {name: [] for name in models}
for target in targets:
    df = data[data['metric_item_label'] == target].dropna(subset=features + ['value']).copy()
    y = (df['value'] > df['value'].median()).astype(int)
    X = df[features].astype(np.float32).values  # Important for NKN

    for name, model in models.items():
        try:
            f1 = cross_val_score(model, X, y, cv=5, scoring=make_scorer(f1_score)).mean()
            model_scores[name].append(f1)
        except Exception as e:
            print(f"⚠️ Model {name} failed on {target}: {e}")
            model_scores[name].append(np.nan)

# Create results DataFrame
results_df = pd.DataFrame(model_scores, index=targets).T
results_df['mean'] = results_df.mean(axis=1)
results_df['sum'] = results_df.sum(axis=1)

plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y=results_df['mean'])
plt.xticks(rotation=45, ha='right')
plt.ylabel("Mean F1 Score")
plt.title("Model Performance Comparison (Mean F1 Score)")
plt.tight_layout()
#plt.savefig("../Output/3.1  Model_Performance_Comparison.png")
plt.show()

# Output summary
print("\n\U0001F4CA Total Performance Summary:")
print(results_df[['mean', 'sum']].sort_values(by='mean', ascending=False))

# Ranking
ranking = results_df['mean'].sort_values(ascending=False).index.tolist()
print("\n\U0001F3C6 Ranked Model Performance:")
top_4_ML=[]
for i, model in enumerate(ranking):
    if i == 0:
        label = '1st'
    elif i == 1:
        label = '2nd'
    elif i == 2:
        label = '3rd'
    else:
        label = f'{i+1}th'
    print(f"{label} Place: {model}")

    if i < 4:
        top_4_ML.append(model)

# Final Results
print(f"Top 4 ML Models: {top_4_ML}")

########################################### Learning Curves ###########################################
# 1. Map Classifiers
class NKN(nn.Module):                                               # Defining NKN
    def __init__(self, input_dim, hidden_dim=32):
        super(NKN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.kernel = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.sin(self.kernel(x))
        return self.out(x)
    
classifier_map = {
    'Logistic Regression': lambda: LogisticRegression(max_iter=1000),
    'Ridge Classifier': lambda: RidgeClassifier(),
    'Random Forest': lambda: RandomForestClassifier(random_state=42),
    'XGBoost': lambda: XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
    'Gradient Boosting': lambda: GradientBoostingClassifier(random_state=42),
    'AdaBoost': lambda: AdaBoostClassifier(),
    'Naive Bayes': lambda: GaussianNB(),
    'SVM (Linear)': lambda: SVC(kernel='linear', probability=True),
    'SVM (RBF)': lambda: SVC(kernel='rbf', probability=True),
    'KNN': lambda: KNeighborsClassifier(),
    'MLP': lambda: MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=300, random_state=42),
    'NKN': lambda: NeuralNetClassifier(NKN, module__input_dim=len(features), max_epochs=20, lr=0.01, optimizer=torch.optim.Adam, verbose=0)
}


# 2. Preprocessing: Encoding & Scaling
for col in features:
    if data[col].dtype == 'object':
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# 3. Using only models, that are in top_4_ML and that exist in the mapping
models = {name: classifier_map[name]() for name in top_4_ML if name in classifier_map}

# Evaluate classifiers
model_scores = {name: [] for name in models}
for target in targets:
    df = data[data['metric_item_label'] == target].dropna(subset=features + ['value']).copy()
    y = (df['value'] > df['value'].median()).astype(int)
    X = df[features]

    if y.nunique() < 2:
        print(f"Skipping {target}: Only one class present in target.")
        continue

    for name, model in models.items():
        try:
            f1 = cross_val_score(model, X, y, cv=5, scoring=make_scorer(f1_score)).mean()
            model_scores[name].append(f1)
        except Exception as e:
            print(f"⚠️ Model {name} failed on {target}: {e}")
            model_scores[name].append(np.nan)


# Additional Metrics
def evaluate_model(model, X, y):
    return {
        'F1-Score': cross_val_score(model, X, y, cv=5, scoring='f1').mean(),
        'Accuracy': cross_val_score(model, X, y, cv=5, scoring='accuracy').mean(),
        'Precision': cross_val_score(model, X, y, cv=5, scoring='precision').mean(),
        'Recall': cross_val_score(model, X, y, cv=5, scoring='recall').mean(),
        'ROC-AUC': cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
    }

print("\nAdvanced Metrics by Target:")
for target in targets:
    df = data[data['metric_item_label'] == target].dropna(subset=features + ['value']).copy()
    y = (df['value'] > df['value'].median()).astype(int)
    X = df[features]

    if y.nunique() < 2:
        continue

    print(f"\n{target}:")
    for name, model in models.items():
        try:
            metrics = evaluate_model(model, X, y)
            print(f"{name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.3f}")
        except Exception as e:
            print(f"⚠️ {name} failed on {target}: {e}")

# Combined Learning Curve Plot
for target in targets:
    df = data[data['metric_item_label'] == target].dropna(subset=features + ['value']).copy()
    y = (df['value'] > df['value'].median()).astype(int)
    X = df[features]

    if y.nunique() < 2:
        continue

    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=5, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 10)
            )
            plt.plot(train_sizes, np.mean(test_scores, axis=1), label=name)
        except Exception as e:
            print(f"⚠️ Learning curve failed for {name} on {target}: {e}")

    plt.title(f'Combined Learning Curve on {target}')
    plt.xlabel('Training Size')
    #plt.xlim(0, 600)
    plt.ylabel('F1 Score')
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../Output/3.1 Combined Learning Curve on {target}.jpg")
    plt.show()

