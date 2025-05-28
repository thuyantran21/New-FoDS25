import pandas as pd

# Load original dataset
data = pd.read_csv("../../Data/BigCitiesHealth.csv")

# Define critical columns for model inputs
critical_columns = ['strata_race_label', 'strata_sex_label', 'geo_strata_poverty', 'geo_strata_Segregation', 'geo_strata_region', 'geo_strata_PopDensity']
targets = ['Lung Cancer Deaths', 'Adult Mental Distress', 'Infant Deaths', 'Life Expectancy', 'High Blood Pressure', 'Low Birthweight', 'Maternal Deaths']


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
