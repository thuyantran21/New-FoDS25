import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# 1. Load data
data = pd.read_csv("../Data/BigCitiesHealth_Cleaned.csv")

# 2. Settings
metrics = ['Lung Cancer Deaths', 'Adult Mental Distress', 
           'Life Expectancy', 'High Blood Pressure', 'Maternal Deaths']

features = ['strata_race_label', 'strata_sex_label', 'geo_strata_poverty',
            'geo_strata_Segregation', 'geo_strata_region', 'geo_strata_PopDensity']

# 3. Label Encoding of Categorical Features
for col in features:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# 4. Define Models
svm_linear = SVC(kernel='linear', probability=True, random_state=42)
svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)

# 5. Evaluation
for metric in metrics:
    print(f"\nMetric: {metric}")
    df = data[data['metric_item_label'] == metric].dropna(subset=features + ['value']).copy()
    y = (df['value'] > df['value'].median()).astype(int)
    X = df[features]

    for name, model in [('SVM (Linear)', svm_linear), ('SVM (RBF)', svm_rbf)]:
        f1 = cross_val_score(model, X, y, cv=5, scoring=make_scorer(f1_score)).mean()
        print(f"{name}: F1-Score = {f1:.4f}")

