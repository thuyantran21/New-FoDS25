

########################################################################################################

from sklearn.svm import SVC
import pandas as pd
import seaborn as sns
import matplotlib as plt

# Lineare SVM
svm_linear = SVC(kernel='linear', probability=True, random_state=42)

# Nichtlineare SVM mit RBF-Kernel
svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)

# Beispielhafte Verwendung für einen Target-Metric
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer

data=pd.read_csv("../Data/BigCitiesHealth.csv")
targets = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths', 'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']
features = ['strata_race_label', 'strata_sex_label', 'geo_strata_poverty', 'geo_strata_Segregation', 'geo_strata_region', 'geo_strata_PopDensity']

comp=[]
# Für ein Beispiel-Target
for targ in targets:
    df = data[data['metric_item_label'] == targ].dropna(subset=features + ['value']).copy()
    y = (df['value'] > df['value'].median()).astype(int)
    X = df[features].astype(float)

    # Bewertung
    for name, model in [('SVM (Linear)', svm_linear), ('SVM (RBF)', svm_rbf)]:
        f1 = cross_val_score(model, X, y, cv=5, scoring=make_scorer(f1_score)).mean()
        print(f"{name} F1-Score for '{targ}': {f1:.4f}")
        comp.append(svm_linear-svm_rbf)
print(f"Comparison - linear-RBF: {comp}")
