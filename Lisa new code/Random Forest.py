import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Daten laden
data = pd.read_csv('../../Data/BigCitiesHealth_Cleaned.csv')  # Ersetze durch deinen Pfad

# Define target metrics (outcome variables)
metrics = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths', 
           'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']

# Define feature set
features = [
    'strata_race_label',
    'strata_sex_label',
    'geo_strata_poverty',
    'geo_strata_Segregation',
    'geo_strata_region',
    'geo_strata_PopDensity'
]

# Datenvorverarbeitung
filtered_data = data[data['metric_item_label'].isin(metrics)]
X = filtered_data[features]
y = filtered_data['metric_item_label']

# One-Hot-Encoding und Skalierung
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Pipeline für den RandomForest
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Grid Search für Hyperparameter-Tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(model, param_grid, cv=cv, verbose=1, n_jobs=-1, scoring='accuracy')

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modelltraining
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Feature Importance (Gesamte Feature-Importance)
rf = best_model.named_steps['classifier']
encoder = best_model.named_steps['preprocessor'].transformers_[1][1]
feature_names = list(numeric_features) + list(encoder.get_feature_names_out(categorical_features))

importances = rf.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot der Feature Importance (Gesamt)
plt.figure(figsize=(12, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance (All Features)')
plt.tight_layout()
plt.savefig("../Outputs/TEST 8. Feature Importance - All Features.png")
plt.show()

# Plot der Subgruppen-Importance (nur für kategoriale Features)
fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

for ax, feature in zip(axes.flat, categorical_features):
    # Filter für Subgruppen dieses Features
    subgroup_importances = importance_df[importance_df['Feature'].str.contains(feature)]
    sns.barplot(data=subgroup_importances, x='Importance', y='Feature', ax=ax, palette='viridis')
    ax.set_title(f'Subgroup Importance for {feature}')
    ax.set_xlabel("Importance")
    ax.set_ylabel("Subgroup")

plt.suptitle('Subgroup Importance for Categorical Features')
plt.tight_layout()
plt.savefig("../Outputs/TEST 8. Subgroup Importance - Categorical Features.png")
plt.show()
