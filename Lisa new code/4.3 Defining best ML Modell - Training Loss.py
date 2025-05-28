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
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# Daten laden
data = pd.read_csv('../Data/BigCitiesHealth.csv')

# Features und Zielmetriken definieren
features = ['strata_race_label', 'strata_sex_label', 'geo_strata_poverty',
            'geo_strata_Segregation', 'geo_strata_region', 'geo_strata_PopDensity']

targets = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths',
           'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']

# Kategorische Variablen encoden
for col in features:
    if data[col].dtype == 'object':
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# Features normalisieren
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Gerätewahl
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modellklassen definieren
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

class NKNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.linear(x))

class NKNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.kernel_layer1 = NKNBlock(input_dim, 32)
        self.kernel_layer2 = NKNBlock(32, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = self.kernel_layer1(x)
        x = self.kernel_layer2(x)
        return self.output(x)

# Ergebnisse speichern
results = []

for target in targets:
    print(f"\n🔍 Evaluating target metric: {target}")
    df = data[data['metric_item_label'] == target].dropna(subset=features + ['value'])
    if df.empty: continue

    X_np = df[features].values
    y_np = df['value'].values

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

    # Modelle definieren
    models = {
        '1st Place: XGBoost': XGBRegressor(random_state=42),
        '2nd Place: Random Forest': RandomForestRegressor(random_state=42),
        '3rd Place: Decision Tree': DecisionTreeRegressor(random_state=42),
        '4th Place: Gradient Boosting': GradientBoostingRegressor(random_state=42),
        '5th Place: KNN': KNeighborsRegressor(),
        '6th Place: AdaBoost': AdaBoostRegressor(random_state=42),
        '7th Place: MLP': MLP(input_dim=X_np.shape[1]).to(device),
        '8th Place: SVM (RBF)': SVR(kernel='rbf'),
        '9th Place: Ridge Classifier': Ridge(),
        '10th Place: Logistic Regression': LinearRegression(),
        '11th Place: SVM (Linear)': SVR(kernel='linear'),
        '13th Place: NKN': NKNModel(input_dim=X_np.shape[1]).to(device)
    }
    # '12th Place: Naive Bayes': GaussianNB(), not included as it's a classification model and does not perform regression
    
    losses_dict = {}
    loss_fn = nn.MSELoss()

    for name, model in models.items():
        if isinstance(model, nn.Module):  # Torch-basierte Modelle
            X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
            y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1).to(device)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001 if 'MLP' in name else 0.01)
            losses = []
            for epoch in range(100):
                optimizer.zero_grad()
                preds = model(X_train)
                loss = loss_fn(preds, y_train)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            losses_dict[name] = losses
        else:  # Sklearn Modelle
            try:
                model.fit(X_train_np, y_train_np)
                preds = model.predict(X_train_np)
                if preds.ndim == 2:
                    preds = preds[:, 0]
                mse = mean_squared_error(y_train_np, preds)
                losses_dict[name] = [mse] * 100  # konstante Linie zur Visualisierung
            except Exception as e:
                losses_dict[name] = [np.nan] * 100
                print(f"⚠️ Fehler bei Modell '{name}': {e}")
        # Evaluiere Modellleistung auf Testdaten und speichere R²
        try:
            if isinstance(model, nn.Module):
                model.eval()
                with torch.no_grad():
                    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
                    y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1).to(device)
                    preds = model(X_test).cpu().numpy()
            else:
                preds = model.predict(X_test_np)
                if preds.ndim == 2:
                    preds = preds[:, 0]

            r2 = r2_score(y_test_np, preds)
            mse = mean_squared_error(y_test_np, preds)
            results.append({
                "Model": name,
                "Metric": target,
                "R2": r2,
                "MSE": mse
            })
        except Exception as e:
            print(f"⚠️ R²-Fehler bei {name} ({target}): {e}")
    # Plot mit 13 Subplots (5 Spalten x 3 Zeilen)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharey=True)
    axes = axes.flatten()

    for idx, (name, losses) in enumerate(losses_dict.items()):
        if idx >= len(axes):
            break
        axes[idx].plot(losses)
        axes[idx].set_title(name)
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("MSE Loss")

    plt.suptitle(f"Model Comparison through Training Loss for {target}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.savefig(f"../Output/4.3 Model_Loss_13Panel_{target}.png")
    #plt.show()

# DataFrame aus vorhandener Ergebnistabelle
results_df = pd.DataFrame(results)  # results muss während des Trainings befüllt worden sein

# Mittelwert-R² pro Modell berechnen
avg_df = results_df.groupby("Model")["R2"].mean().sort_values(ascending=False)

# Einfacher Barplot zur Modellvergleich
plt.figure(figsize=(14, 8))
avg_df.plot(kind='bar')
plt.ylabel("Durchschnittlicher R²")
plt.ylim(0, 1)
plt.title("Modell-Performance über alle Metriken (Durchschnittliches R²)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("../Output/4.3 Model_Summary_Performance_Barplot.png")
plt.show()
