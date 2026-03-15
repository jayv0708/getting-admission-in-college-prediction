import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("Loading data...")
# Loading the dataset
df = pd.read_csv('admission_predict.csv')

# Renaming the columns with appropriate names
df = df.rename(columns={'GRE Score': 'GRE', 'TOEFL Score': 'TOEFL', 'LOR ': 'LOR', 'Chance of Admit ': 'Probability'})

# Removing the serial no, column
df.drop('Serial No.', axis='columns', inplace=True)

# Replacing the 0 values from ['GRE','TOEFL','University Rating','SOP','LOR','CGPA'] by NaN
df_copy = df.copy(deep=True)
df_copy[['GRE','TOEFL','University Rating','SOP','LOR','CGPA']] = df_copy[['GRE','TOEFL','University Rating','SOP','LOR','CGPA']].replace(0, np.NaN)

# Splitting the dataset in features and label
X = df_copy.drop('Probability', axis='columns')
y = df_copy['Probability']

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model Accuracy (R^2 Score): {score*100:.2f}%")

print("\n--- Predictions ---")
# Prediction 1
# Input in the form : GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research
pred1 = round(model.predict([[337, 118, 4, 4.5, 4.5, 9.65, 0]])[0]*100, 3)
print(f'Chance of getting into UCLA (Profile 1) is {pred1}%')

# Prediction 2
pred2 = round(model.predict([[320, 113, 2, 2.0, 2.5, 8.64, 1]])[0]*100, 3)
print(f'Chance of getting into UCLA (Profile 2) is {pred2}%')
