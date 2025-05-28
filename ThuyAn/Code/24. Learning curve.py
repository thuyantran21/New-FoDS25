import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


# Load dataset (cleaned)
data = pd.read_csv("../../Data/BigCitiesHealth.csv")
df = pd.read_excel("../Outputs/ML_Model_Results.xlsx")

