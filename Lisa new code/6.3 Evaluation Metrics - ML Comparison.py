import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Daten laden
data = pd.read_csv('../../Data/BigCitiesHealth.csv')

# Features und Zielmetrik definieren
features = ['strata_race_label', 'strata_sex_label', 'geo_strata_poverty',
            'geo_strata_Segregation', 'geo_strata_region', 'geo_strata_PopDensity']

all_metrics = ['Life Expectancy']

# Auswahl der potenziellen Einflussmetriken
metric_columns = data['metric_item_label'].unique()
metric_data = data.pivot_table(index=['geo_label_city', 'geo_label_state'],
                                columns='metric_item_label', values='value', aggfunc='mean')
metric_data = metric_data.dropna(axis=1)  # Nur vollständige Metriken behalten

# Ziel und Merkmale festlegen
X = metric_data.drop(columns=all_metrics, errors='ignore')
y = metric_data[all_metrics[0]]

# Normalisieren
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Torch Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MLP Modell
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# Modell-Liste
models = {
    '1st Place: XGBoost': XGBRegressor(random_state=42),
    '2nd Place: Random Forest': RandomForestRegressor(random_state=42),
    '3rd Place: Decision Tree': DecisionTreeRegressor(random_state=42),
    '4th Place: Gradient Boosting': GradientBoostingRegressor(random_state=42),
    '5th Place: KNN': KNeighborsRegressor(),
    '6th Place: AdaBoost': AdaBoostRegressor(random_state=42),
    '7th Place: MLP': MLP(input_dim=X.shape[1]).to(device),
    '8th Place: SVM (RBF)': SVR(kernel='rbf'),
    '9th Place: Ridge Classifier': Ridge(),
    '10th Place: Logistic Regression': LinearRegression(),
    '11th Place: SVM (Linear)': SVR(kernel='linear'),
    '13th Place: NKN': MLP(input_dim=X.shape[1]).to(device)  # NKN als Platzhalter durch MLP ersetzt
}

results = []
feature_importances = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    if isinstance(model, nn.Module):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
        for epoch in range(100):
            optimizer.zero_grad()
            preds = model(X_train_tensor)
            loss = loss_fn(preds, y_train_tensor)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            preds = model(X_test_tensor).cpu().numpy()
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        if hasattr(model, 'feature_importances_'):
            feature_importances[name] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importances[name] = np.abs(model.coef_)

    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    results.append({'Model': name, 'R2': r2, 'MSE': mse})

# Ergebnisse darstellen
results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
print("\nModellvergleich zur Einschätzung des Einflusses auf Life Expectancy:")
print(results_df)

# Wichtigkeit der Merkmale (sofern vorhanden) visualisieren
for name, importances in feature_importances.items():
    sorted_idx = np.argsort(importances)[::-1]
    top_features = X.columns[sorted_idx][:10]
    top_values = importances[sorted_idx][:10]
    plt.figure(figsize=(10, 5))
    plt.barh(top_features[::-1], top_values[::-1])
    plt.title(f"Top Einflussfaktoren laut {name}")
    plt.tight_layout()
    plt.savefig(f"../Output/5.0 FeatureImportance_{name}.png")
    plt.show()
