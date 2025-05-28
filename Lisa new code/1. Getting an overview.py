#### Getting a brief overview of the Data Set ####

import pandas as pd

data = pd.read_csv("../../Data/BigCitiesHealth.csv")


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
missing_summary.to_excel("../Output/1. Data_Exploration_Summary.xlsx", sheet_name="Missing Data Details", index=False)
print("\nData exploration results saved to: Data_Exploration_Summary.xlsx")

# Features
print("To explore Research Question following features are observed: ")
features =  ("strata_race_label", 
             "strata_sex_label", 
             "geo_strata_poverty", 
             "geo_strata_region", 
             "geo_strata_PopDensity"
             )
print(features)

# Target Metrics
metrics  = [
    'Infant Deaths', 
    'Life Expectancy',
    'Low Birthweight',
    'Adult Mental Distress',
    'High Blood Pressure',
    'Lung Cancer Deaths',
    'Maternal Deaths'
]

print("\n################## Metrics ###################")

# All Metrics
metrics = data["metric_item_label"].unique()
print(metrics)


