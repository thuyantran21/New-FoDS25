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
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor

#######################################################################################################
########################################### General Settings ##########################################
#######################################################################################################
# Daten laden
data = pd.read_csv('../Data/BigCitiesHealth.csv')

# Features und Zielmetriken definieren
features = ['strata_race_label', 'strata_sex_label', 'geo_strata_poverty',
            'geo_strata_Segregation', 'geo_strata_region', 'geo_strata_PopDensity']

targets = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths',
           'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']

top_4_ML = ['XGBoost', 'Random Forest', 'Gradient Boosting', 'KNN']

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


#######################################################################################################
########################################### Training Loss Curves ######################################
#######################################################################################################
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

    train_losses_dict = {}
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

        train_losses_dict[name] = losses
    
    # 5. Plot mit allen Modellen
    plt.figure(figsize=(10, 6))
    for name, losses in train_losses_dict.items():
        plt.plot(range(1, len(losses) + 1), losses, label=name)

    plt.title(f"Training Loss Comparison - {target}")
    plt.xlabel("Iterations")
    plt.xlim(0, 100)
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../Output/3.3 Combined Loss Comparison - {target}.png")
    #plt.show()


#####################################################################################################
#                                            Validation Loss                                        #
#####################################################################################################
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
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Ridge Classifier': Ridge(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(eval_metric='logloss', random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(),
        'AdaBoost': AdaBoostRegressor(),
        'Naive Bayes': GaussianNB(),  # Nur für Klassifikation sinnvoll
        'SVM (Linear)': SVR(kernel='linear'),
        'SVM (RBF)': SVR(kernel='rbf'),
        'KNN': KNeighborsRegressor(),
        'MLP': MLP(input_dim=len(features)).to(device),  # deine eigene torch.nn.Module Klasse
        'NKN': NKNModel(input_dim=len(features)).to(device)  # ebenfalls torch.nn.Module
    }
    model_wrappers = regressor_map

# Validation Loss
n_iterations = 50
for target in targets:
    print(f"\n🔍 Processing: {target}")
    df = data[data['metric_item_label'] == target].dropna(subset=features + ['value'])
    if df.empty:
        print(f"⚠️ Skipping {target}, no valid data.")
        continue

    X = df[features].values
    y = df['value'].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

    plt.figure(figsize=(12, 7))

    val_losses_dict = {}
    for name in top_4_ML:
        val_losses = []

        if name in model_wrappers:
            for i in range(1, n_iterations + 1):
                try:
                    model = model_wrappers[name](i)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    val_losses.append(mean_squared_error(y_val, preds))
                except Exception as e:
                    val_losses.append(np.nan)
                    print(f"⚠️ {name} failed at iteration {i}: {e}")
        else:
            try:
                model = models[name]
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                val_loss = mean_squared_error(y_val, preds)
                val_losses = [val_loss] * n_iterations  # konstante Linie
            except Exception as e:
                val_losses = [np.nan] * n_iterations
                print(f"⚠️ {name} failed once: {e}")
        val_losses_dict[name] = val_losses 
        plt.plot(range(1, n_iterations + 1), val_losses, label=name)

    plt.title(f"Validation Loss Curve – {target}")
    plt.xlabel("Iterations")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../Output/3.3 Validation Loss - {target}.png")
    #plt.show()


#######################################################################################################
########################################## Comparing Train vs. Val ####################################
#######################################################################################################
threshold = 0.10    # 10% Toleranz
threshold2 = 0.20   # 20% Grenze
for model_name in top_4_ML:
    train_loss = train_losses_dict[model_name][-1]  # Letzter Wert (nach n Iterationen)
    val_loss = val_losses_dict[model_name][-1]

    # Verhältnis vergleichen
    relative_diff = abs(train_loss - val_loss) / val_loss

    print(f"\nModel: {model_name}")
    print(f"  Final Training Loss:   {train_loss:.4f}")
    print(f"  Final Validation Loss: {val_loss:.4f}")
    print(f"  Relative Difference:   {relative_diff*100:.2f}%")

    if relative_diff <= threshold:
        print("✅ Training and Validation Loss match closely.")
        print("✅ Good generalization. Low overfitting risk.")
    
    elif relative_diff <= threshold2 and relative_diff >= threshold:
        print("⚠️ Training and Validation Loss show small derivations.")
        print("⚠️ Good generalization. But beginning overfitting risk.")

    else:
        print("❌ Significant gap between training and validation.")
        print("❌ Possible overfitting or underfitting.")
