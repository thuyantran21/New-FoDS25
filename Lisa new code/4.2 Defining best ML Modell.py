import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from skorch import NeuralNetClassifier
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


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
    'AdaBoost': AdaBoostClassifier(),                                   #'Decision Tree is pretty similar to Random Forest and therefor was not included
    'Linear Regression': LinearRegression(),
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
#plt.savefig("../Output/Model_Performance_Comparison.png")
#plt.show()

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
print(f"Top 4 ML Models: {top_4_ML}")


# 1. Map Klassifikatoren-Namen auf passende Regressor-Klassen
regressor_map = {
    'XGBoost': lambda i=None: XGBRegressor(n_estimators=i or 100, learning_rate=0.1, random_state=42),
    'Random Forest': lambda i=None: RandomForestRegressor(n_estimators=i or 100, random_state=42),
    'Gradient Boosting': lambda i=None: GradientBoostingRegressor(n_estimators=i or 100, learning_rate=0.1, random_state=42),
    'Linear Regression': lambda i=None: LinearRegression(),
    'Ridge Classifier': lambda i=None: Ridge(),
    'SVM (Linear)': lambda i=None: SVR(kernel='linear'),
    'SVM (RBF)': lambda i=None: SVR(kernel='rbf'),
    'Naive Bayes': lambda i=None: GaussianNB(),  # Kein Regressor, aber für Demo bleibt er konstant
    'Decision Tree': lambda i=None: DecisionTreeRegressor(random_state=42),
    'MLP': lambda i=None: MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=300, random_state=42),
    'KNN': lambda i=None: KNeighborsRegressor(n_neighbors=5),
    'NKN': lambda i=None: MLPRegressor(hidden_layer_sizes=(50,), max_iter=100, random_state=42)  # Dummy-NN als Platzhalter
}

# 2. Preprocessing: Encoding & Skalierung
for col in features:
    if data[col].dtype == 'object':
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# 3. Nutze NUR Modelle, die in den Top 4 auftauchen und im Mapping existieren
model_dict = {name: regressor_map[name]() for name in top_4_ML if name in regressor_map}

# 4. Trainingskurven plotten für jede Zielmetrik
for target in targets:
    print(f"\n📊 Processing metric: {target}")
    df = data[data['metric_item_label'] == target].dropna(subset=features + ['value'])

    X = df[features]
    y = df['value'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    losses_dict = {}
    for name in model_dict.keys():
        losses = []

        # Modelle ohne n_estimators
        if name not in ['XGBoost', 'Random Forest', 'Gradient Boosting']:
            model = regressor_map[name]()
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_train)
                mse = mean_squared_error(y_train, preds)
                losses = [mse] * 100  # konstante Linie für Vergleich
            except Exception as e:
                print(f"⚠️ Fehler bei {name}: {e}")
                continue
        else:
            # Modelle mit n_estimators iterativ trainieren
            for i in range(1, 101):
                model = regressor_map[name](i)
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_train)
                    mse = mean_squared_error(y_train, preds)
                    losses.append(mse)
                except Exception as e:
                    print(f"⚠️ Fehler bei {name} mit {i} estimators: {e}")
                    losses.append(np.nan)

        losses_dict[name] = losses

    # 5. Plot mit allen Modellen
    plt.figure(figsize=(10, 6))
    for name, losses in losses_dict.items():
        plt.plot(range(1, len(losses) + 1), losses, label=name)

    plt.title(f"Training Loss Comparison - {target}")
    plt.xlabel("Iterations")
    plt.xlim(0, 100)
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../Output/4.2 Combined_LossComparison_{target}.png")
    plt.show()

