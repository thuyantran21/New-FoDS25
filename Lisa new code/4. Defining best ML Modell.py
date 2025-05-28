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

# 2. Define features and target
#features = ['strata_race_label', 'strata_sex_label', 'geo_strata_poverty', 'geo_strata_region', 'geo_strata_PopDensity']
features = ['strata_race_label', 'strata_sex_label', 'geo_strata_poverty', 'geo_strata_Segregation', 'geo_strata_region', 'geo_strata_PopDensity']
targets = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths', 'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']
#targets = ['Cardiovascular Disease Deaths','Diabetes Deaths','Injury Deaths','All Cancer Deaths']

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
plt.savefig("../Output/4. Defining best Model - Model Comparison by F1-Scores.jpg")
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
    plt.savefig(f"../Output/4. Defining best Model - Learning Curve for {title}.jpg")
    plt.show()

for name, model in models.items():
    plot_learning_curve(model, X, y, name)

#################################   FÜR ZUKUNFT:     EVTL. HEATMAP HINZUFÜGEN!   ##########################################

# 10. Linearity Check (Boxplots)
plt.figure(figsize=(15, 8))
sns.boxplot(data=df_model[features])
plt.title('Boxplot of Features')
plt.savefig("../Output/4. Linearity Check - Features by Boxplots.jpg")
plt.show()





########################################################################################################
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('../../Data/BigCitiesHealth.csv')

# Define inputs and targets
features = ['strata_race_label', 'strata_sex_label', 'geo_strata_poverty',
            'geo_strata_Segregation', 'geo_strata_region', 'geo_strata_PopDensity']
targets = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths',
           'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']

# Preprocess
label_encoders = {}
for col in features:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

# Normalize features
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Loop over each health metric
for target_metric in targets:
    print(f"\nProcessing metric: {target_metric}")
    df = data[data['metric_item_label'] == target_metric].dropna(subset=features + ['value'])

    X = torch.tensor(df[features].values, dtype=torch.float32)
    y = torch.tensor(df['value'].values, dtype=torch.float32).unsqueeze(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a simple NKN-like model
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

    model = NKNModel(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    losses = []
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        score = r2_score(y_test.numpy(), y_pred.numpy())
        print(f"R² Score for {target_metric}: {score:.4f}")

    # Plot training loss
    plt.plot(losses)
    plt.title(f"Training Loss - {target_metric}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig(f"../Training Loss - {target_metric}")
    plt.show()


#######################################################################################################
############################################# MLP REGRESSION ##########################################
#######################################################################################################
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Daten laden
data = pd.read_csv('../../Data/BigCitiesHealth.csv')

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

# Gerät wählen (GPU falls verfügbar)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelldefinition
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

# Für jede Metrik trainieren und evaluieren
for target in targets:
    print(f"\n🔍 Processing metric: {target}")
    df = data[data['metric_item_label'] == target].dropna(subset=features + ['value'])

    # Eingabe- und Zielvektoren
    X = torch.tensor(df[features].values, dtype=torch.float32)
    y = torch.tensor(df['value'].values, dtype=torch.float32).unsqueeze(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # DataLoader
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)

    # Modell, Loss, Optimizer
    model = MLP(input_dim=X.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    model.train()
    losses = []
    for epoch in range(100):
        epoch_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_dl))
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
        y_test_np = y_test.numpy()
        r2 = r2_score(y_test_np, preds)
        mse = mean_squared_error(y_test_np, preds)
        print(f"R² Score: {r2:.4f}, MSE: {mse:.2f}")

    # Trainingskurve plotten
    plt.plot(losses)
    plt.title(f"Loss Curve - {target}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../Output/MLP_LossCurve_{target}.png")
    plt.show()
