import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
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
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# Load data
data = pd.read_csv('../../Data/BigCitiesHealth.csv')
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



# Kategorische Variablen encoden
for col in features:
    if data[col].dtype == 'object':
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# Features normalisieren
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Modelle vorbereiten
model_dict = {
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression()
}

# Trainingskurven speichern und plotten
for target in targets:
    print(f"\nProcessing metric: {target}")
    df = data[data['metric_item_label'] == target].dropna(subset=features + ['value'])

    X = df[features].values
    y = df['value'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    losses_dict = {}
    for name, model in model_dict.items():
        losses = []
    for name, model in model_dict.items():
        if name == 'KNN':
            # Optional: konstante Linie, da kein n_estimators
            model.fit(X_train, y_train)
            preds = model.predict(X_train)
            mse = mean_squared_error(y_train, preds)
            losses = [mse] * 100  # Konstante Kurve für Vergleich
        else:
            for i in range(1, 101):
                if name == 'XGBoost':
                    temp_model = XGBRegressor(n_estimators=i, learning_rate=0.1, random_state=42)
                elif name == 'Random Forest':
                    temp_model = RandomForestRegressor(n_estimators=i, random_state=42)
                elif name == 'Gradient Boosting':
                    temp_model = GradientBoostingClassifier(n_estimators=i, learning_rate=0.1, random_state=42)
                
                temp_model.fit(X_train, y_train)
                preds = temp_model.predict(X_train)
                mse = mean_squared_error(y_train, preds)
                losses.append(mse)
        
        losses_dict[name] = losses

    # Plot mit allen drei Modellen in einem Diagramm
    plt.figure(figsize=(10, 6))
    for name, losses in losses_dict.items():
        plt.plot(range(1, len(losses) + 1), losses, label=name)

    plt.title(f"Training Loss Comparison - {target}")
    plt.xlabel("Iterations")
    plt.xlim(0,100)
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../Output/4.2 Combined_LossComparison_{target}.png")
    plt.show()

