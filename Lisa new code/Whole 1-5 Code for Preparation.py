import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import f1_score, make_scorer, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy.stats import f_oneway, pearsonr, spearmanr


#########################################################################################################################
################################################## General Settings #####################################################
#########################################################################################################################

# Load dataset (cleaned)
data = pd.read_csv("../../Data/BigCitiesHealth.csv")

# Metrics and Features (edit for other research questions!)
features = [
    'strata_race_label',
    'strata_sex_label',
    'geo_strata_poverty',
    'geo_strata_Segregation',
    'geo_strata_region',
    'geo_strata_PopDensity',
    'geo_strata_Population'
]

metrics = data['metric_item_label'].unique().tolist()            # All Metrics

targets = metrics

#########################################################################################################################
################################################## 1. Getting an Overview ###############################################
#########################################################################################################################

# Exploring basics of Data Set
print("################## Basics of original Data Set ###################")
n_rows = data.shape[0]
n_columns = data.columns.shape[0]

print("Number of Rows: ", n_rows)
print("Number of Columns: ", n_columns)

# Missing Data
print("################## Missing Data ###################")
missing_summary = data.isnull().sum().reset_index()
missing_summary.columns = ['Column', 'Missing Values']
# print(missing_summary)

# Sum of missing Data
missing_sum = data.isna().sum().sum()
print("Sum of missing Data: ", missing_sum) 

# Save missing data directly to Excel
missing_summary.to_excel("../Outputs/1. Data_Exploration_Summary.xlsx", sheet_name="Missing Data Details", index=False)
print("\nData exploration results saved to: Data_Exploration_Summary.xlsx")

print("\n################## All Metrics ###################")

# All Metrics
metrics = data["metric_item_label"].unique()
print(metrics)


#########################################################################################################################
################################################## 2. Metric Selection ##################################################
#########################################################################################################################

# Alle Gesundheitsmetriken
all_metrics = data['metric_item_label'].unique().tolist()
# all_metrics =  ['COVID-19 Deaths', 'Diabetes Deaths', 'Walking to Work', 'Deaths from All Causes', 'Life Expectancy', 'Uninsured, All Ages', 'Uninsured, Child', 'Dental Care', 'Prenatal Care', 'People with Disabilities', 'Teen Asthma', 'All Cancer Deaths', 'Breast Cancer Deaths', 'Lung Cancer Deaths', 'Cardiovascular Disease Deaths', 'Heart Disease Deaths', 'High Blood Pressure', 'Diabetes', 'Adult Obesity', 'Teen Obesity', 'Adult Physical Inactivity', 'Teen Physical Activity Levels', 'Teen Physical Inactivity', 'Teen Physical Education', 'Teen Computer Time', 'Teen TV Time', 'Teen Soda', 'Teen Breakfast', 'Flu Vaccinations, Medicare', 'Pneumonia or Influenza Deaths', 'New Tuberculosis Cases', 'HIV-Related Deaths', 'HIV/AIDS Prevalence', 'New Chlamydia Cases', 'Syphilis Prevalence', 'Syphilis, Newborns', 'New Gonorrhea Cases', 'Maternal Deaths', 'Infant Deaths', 'Low Birthweight', 'Teen Births', 'Teen Birth Control', 'Opioid Overdose Deaths', 'Adult Binge Drinking', 'Drug Overdose Deaths', 'Adult Smoking', 'Teen Smoking', 'Teen Alcohol', 'Teen Marijuana', 'Adult Mental Distress', 'Suicide', 'Teen Mental Distress', 'Teen Suicidal Ideation', 'Electronic Bullying', 'School Bullying', 'Child Lead Testing', 'Child Lead Levels 5+ mcg/dL', 'Child Lead Levels 10+ mcg/dL', 'Housing Lead Risk', 'Injury Deaths', 'Firearm Deaths', 'Motor Vehicle Deaths', 'Police Killings', 'Racial Disparity in Police Killings', 'Violent Crime', 'Homicides', 'Weapons in School', 'Fighting in School', 'Vacant Housing Units', 'Owner Occupied Housing', 'Renters vs. Owners', 'Preschool Enrollment', 'College Graduates', 'Poverty in All Ages', 'Poverty and Near Poverty in All Ages', 'Poverty in Children', 'Per-capita Household Income', 'Households with Higher-Incomes', 'Public Assistance', 'Unemployment', 'Service Workers', 'Excessive Housing Cost', 'Household Income Inequality', 'Income Inequality', 'Racial Segregation, White and non-White', 'Racial Segregation, White and Black', 'Racial Segregation, White and Asian', 'Racial Segregation, White and Hispanic', 'Limited Supermarket Access', 'Riding Bike to Work', 'Premature Death', 'Lack of Car', 'Public Transportation Use', 'Drives Alone to Work', 'Longer Driving Commute Time', 'Poor Air Quality', 'Hazardous Air Quality', 'Single-Parent Families', 'Population Density', 'Children', 'Seniors', 'Minority Population', 'Primarily Speak English', 'Primarily Speak Chinese', 'Primarily Speak Spanish', 'Foreign Born Population', 'Homelessness, Non-Whites', 'Homelessness, Children', 'Homelessness and Vacant Housing']
# print("All metrics: ", metrics)

# Nur Gesundheitsmetriken
health_metrics = ['COVID-19 Deaths', 'Diabetes Deaths', 'Deaths from All Causes', 'Life Expectancy','Teen Asthma', 'All Cancer Deaths', 'Breast Cancer Deaths', 'Lung Cancer Deaths', 'Cardiovascular Disease Deaths', 'Heart Disease Deaths', 'High Blood Pressure', 'Diabetes', 'Adult Obesity', 'Teen Obesity', 'Pneumonia or Influenza Deaths', 'New Tuberculosis Cases', 'HIV-Related Deaths', 'HIV/AIDS Prevalence', 'New Chlamydia Cases', 'Syphilis Prevalence', 'Syphilis', 'New Gonorrhea Cases', 'Maternal Deaths', 'Infant Deaths', 'Low Birthweight','Adult Mental Distress', 'Suicide', 'Teen Mental Distress', 'Teen Suicidal Ideation']

print("Total of Health Metrics: ", len(health_metrics))      # Result:   Total of 29 health metrics


relevant_metrics = [
    'Diabetes Deaths', 'Life Expectancy',                                           #'Deaths from All Causes' includes accidents, therefore not part of this list!
    'All Cancer Deaths', 'Breast Cancer Deaths', 'Lung Cancer Deaths', 
    'Cardiovascular Disease Deaths', 'Heart Disease Deaths', 'High Blood Pressure', 
    'Diabetes', 'Pneumonia or Influenza Deaths', 'Maternal Deaths', 'Infant Deaths', 
    'Low Birthweight', 'Adult Mental Distress', 'Teen Mental Distress'           # Obesity is a Riskfactor, not a Metric
    ]
print("Total of Health Metrics: ", len(relevant_metrics))      # Result:   Total of 16 health metrics

# Überprüfen, ob die Features existieren
missing_features = [f for f in features if f not in data.columns]
if missing_features:
    print(f"Fehler: Die folgenden Features fehlen im Datensatz: {missing_features}")
else:
    # Daten filtern
    filtered_data = data[data['metric_item_label'].isin(relevant_metrics)]

    if filtered_data.empty:
        print("Fehler: Keine relevanten Gesundheitsmetriken im Datensatz.")
    else:
        # Signifikanzanalyse (ANOVA, Pearson, Spearman)
        stat_results = {}
        for metric in relevant_metrics:
            metric_data = filtered_data[filtered_data['metric_item_label'] == metric]

            if not metric_data.empty:
                groups = []
                for feature in features:
                    for value in metric_data[feature].dropna().unique():
                        group_values = metric_data[metric_data[feature] == value]['value'].dropna().values
                        if len(group_values) > 1:
                            groups.append(group_values)

                if len(groups) > 1:
                    try:
                        f_stat, p_value_anova = f_oneway(*groups)
                    except ValueError:
                        p_value_anova = np.nan
                else:
                    p_value_anova = np.nan

                # Pearson und Spearman Korrelationsanalyse
                if 'Life Expectancy' in relevant_metrics:
                    life_expectancy = filtered_data[filtered_data['metric_item_label'] == 'Life Expectancy']['value'].dropna().values
                    target_values = metric_data['value'].dropna().values

                    # Länge beider Arrays angleichen
                    min_length = min(len(target_values), len(life_expectancy))
                    target_values = target_values[:min_length]
                    life_expectancy = life_expectancy[:min_length]

                    if len(target_values) > 1:
                        pearson_corr = np.corrcoef(target_values, life_expectancy)[0, 1]
                        spearman_corr, _ = spearmanr(target_values, life_expectancy)
                    else:
                        pearson_corr = np.nan
                        spearman_corr = np.nan
                else:
                    pearson_corr = np.nan
                    spearman_corr = np.nan

                stat_results[metric] = {
                    'ANOVA p-value': p_value_anova,
                    'Pearson Correlation': pearson_corr,
                    'Spearman Correlation': spearman_corr
                }

        # Ergebnisse anzeigen
        stat_df = pd.DataFrame(stat_results).T
        print("\nSignifikanzanalyse der Gesundheitsmetriken:")
        print(stat_df)

        # Signifikante und stark korrelierte Features
        significant_features = stat_df[stat_df['ANOVA p-value'] < 0.05].index.tolist()
        stat_df['Highest Correlation'] = stat_df[['Pearson Correlation', 'Spearman Correlation']].abs().max(axis=1).fillna(0)
        top_10_high_corr = stat_df['Highest Correlation'].nlargest(7).index.tolist()

        significant_and_correlated = list(set(significant_features) & set(top_10_high_corr)) # deadliest features

        print("\nSignifikante Features:", significant_features)
        print("\nStark korrelierte Features:", top_10_high_corr)

        # Visualisierung der Signifikanz
        plt.figure(figsize=(12, 8))
        if not stat_df.empty:
            sns.heatmap(stat_df.dropna().T, cmap='coolwarm', annot=True)
        else:
            print("Keine signifikanten Ergebnisse für Pearson und Spearman.")

        plt.title("Significance Analysis of Health Metrics")
        plt.savefig("../Outputs/2. Statistical Correlation of Metrics.jpg")
        plt.show()

        # Speicherung der Ergebnisse
        stat_df.to_excel("../Outputs/2. Health_Metrics_Significance_Analysis.xlsx")


# Metrics for future codes:
metrics = significant_and_correlated
print(f"\nFinal Metric Selection:  \n{metrics}")

# Small Check: print(len(metrics))
#########################################################################################################################
################################################## 3. Feature Selection #################################################
#########################################################################################################################

targets = metrics

# Speichere ANOVA-Ergebnisse für alle Features
anova_results_all_features = []
anova_results_per_target = []
sign_feature = []

# Zählt, wie oft jedes Feature signifikant ist
feature_significance_count = {feature: 0 for feature in features}

for target in targets:
    print(f"\nProcessing Metric: {target}")
    df_metric = data[data['metric_item_label'] == target].dropna(subset=features + ['value'])

    # ANOVA für alle Features
    for feature in features:
        groups = [group['value'].values for _, group in df_metric.groupby(feature) if len(group) > 1]
        if len(groups) > 1:
            f_stat, p_value = f_oneway(*groups)
            significance = "Yes" if p_value < 0.05 else "No"
            anova_results_per_target.append({
                'Metric': target,
                'Feature': feature,
                'F-Statistic': round(f_stat, 4),
                'p-value': round(p_value, 4),
                'Significant': significance
            })
            
            # Zählt Signifikanz für jedes Feature
            if p_value < 0.05:
                feature_significance_count[feature] += 1



# Ergebnisse in einem DataFrame speichern
anova_df_per_target = pd.DataFrame(anova_results_per_target)

# Sicherstellen, dass der Pfad existiert
output_path = os.path.join(os.getcwd(), "../Outputs/3. ANOVA_Results_Comparison.xlsx")

with pd.ExcelWriter(output_path) as writer:
    anova_df_per_target.to_excel(writer, sheet_name="ANOVA_Per_Target", index=False)

#print(f"\nANOVA-Ergebnisse erfolgreich in {output_path} gespeichert.")


# Lade die ANOVA-Ergebnistabelle aus der hochgeladenen Excel-Datei
anova_df = pd.read_excel(output_path, sheet_name="ANOVA_Per_Target")

# Identifiziere die Features, die in allen Zeilen "Yes" für "Significant" haben
features = []


# Überprüfe für jedes Feature, ob es maximal ein "No" für "Significant" hat
for feature in anova_df['Feature'].unique():
    no_count = anova_df[(anova_df['Feature'] == feature) & (anova_df['Significant'] == 'No')].shape[0]
    
    # Nur hinzufügen, wenn es maximal ein "No" gibt
    if no_count <= 1:
        features.append(feature)

# Ausgabe
print(f"\nFinal Feature Selection: \n{features}")


#########################################################################################################################
#################################################### Zwischenstand der Code-Resultate ###############################################
#########################################################################################################################
print(f"\n########### Brief Overview on Code Results #############")
print(f"Final Metric Selection:  \n{metrics}")
print(f"Final Feature Selection: \n{features}")

#########################################################################################################################
#################################################### 4. Defining ML Model ###############################################
#########################################################################################################################

targets = sorted(metrics)
print("Sorted targets:", targets)


# Check if target columns exist
missing_targets = [t for t in targets if t not in data['metric_item_label'].unique()]
if missing_targets:
    print(f"\nError: The following target columns are missing: {missing_targets}")
    print("\nAvailable metrics in the dataset:")
    print(data['metric_item_label'].unique())
    exit()

# 3. Prepare data for each 

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
plt.savefig("../Outputs/4. Defining best Model - Model Comparison by F1-Scores.jpg")
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
    plt.xlim(0,600)
    plt.ylabel('F1-Score')
    plt.legend()
    plt.savefig(f"../Outputs/4. Defining best Model - Learning Curve for {title}.jpg")
    plt.show()

for name, model in models.items():
    plot_learning_curve(model, X, y, name)

#################################   FÜR ZUKUNFT:     EVTL. HEATMAP HINZUFÜGEN!   ##########################################

# 10. Linearity Check (Boxplots)
plt.figure(figsize=(15, 8))
sns.boxplot(data=df_model[features])
plt.title('Boxplot of Features')
plt.savefig("../Outputs/4. Linearity Check - Features by Boxplots.jpg")
plt.show()

#########################################################################################################################
############################################ 5. Handling Missing Data ###################################################
#########################################################################################################################

# Define critical columns for model inputs
critical_columns = features
# Initial data overview
initial_rows = data.shape[0]

# Calculate missing values before filtering
missing_summary = data[critical_columns].isnull().sum().reset_index()
missing_summary.columns = ['Column', 'Missing Values']
missing_summary['% of Total Rows'] = (missing_summary['Missing Values'] / initial_rows) * 100

print("\n#####################################   Initial Missing Values Summary   #####################################\n")
print(missing_summary)

unique_missing_rows = data[critical_columns].isnull().any(axis=1).sum()
print("Amount of rows with missing Data in critical columns: ", unique_missing_rows)

# Filter the data by removing rows with missing values in critical columns
data_clean = data.dropna(subset=critical_columns)
cleaned_rows = data_clean.shape[0]

# Calculate rows removed
rows_removed = initial_rows - cleaned_rows
percentage_removed = (rows_removed / initial_rows) * 100

print("\n#####################################         Results of cleaning        #####################################")
print(f"Total Rows Before Filtering: {initial_rows}")
print(f"Total Rows After Filtering: {cleaned_rows}")
print(f"Total Rows Removed: {rows_removed} ({percentage_removed:.2f}%)")

# Save cleaned data for ML Model
data_clean.to_csv("../../Data/BigCitiesHealth_Cleaned.csv", index=False)


print("\n#####################################        Checking new Data Set       #####################################") 
# Load dataset (cleaned)
data2 = pd.read_csv("../../Data/BigCitiesHealth_Cleaned.csv")

# Exploring basics of Data Set
print("Basics of Cleaned Data Set:")
n_rows = data2.shape[0]
n_columns = data2.columns.shape[0]

print("The Cleaned Data Set has", n_rows, "Rows and ", n_columns, "Columns.")
#print("Number of Columns: ", n_columns)


# Missing Data
print("\nMissing Data:")
missing_summary = data2.isnull().sum().reset_index()
missing_summary.columns = ['Column', 'Missing Values']
print(missing_summary)

# Sum of missing Data
missing_sum = data2.isna().sum().sum()
print("\t\t        Overall Sum\t    ", missing_sum)         # Frage: Anfügen an vorherige Liste?

# Control of new Data Set BigCitiesHealth_Cleaned.csv
print("\nChecking the new Data Set")
# Check 1
if n_rows == cleaned_rows:
    print("Check 1:   Amount of rows is accurate!")
# Check 2
if data2[critical_columns].isnull().any(axis=1).sum() == 0:
    print("Check 2:   Amount of missing values is Zero." )
    print("Result:    Data Set was well cleaned, no missing values in features left! ")
    print("                                 . .     . .     . .           ")
    print("                                  U       U       U             ")

