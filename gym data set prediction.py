import pandas as pd
import numpy as np
import  matplotlib
from matplotlib import pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing._label import LabelEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
sns.set(color_codes = True)
df=pd.read_csv("data set\gym.csv")
print(df.shape)
pd.set_option('display.max_column', None)
print(df.dtypes)
df = df.drop("Gender", axis = 1)
df = df.drop("Workout_Type",axis = 1)
print((df.head(5)))
print(df.shape)
duplicate = df[df.duplicated()]
df = df.drop_duplicates()
print(duplicate.shape)
print(df.isnull().sum())
target = df['Calories_Burned']
features= df.drop(columns = ['Calories_Burned'])
# SPLITING THE DATA SET INTO TRAINING AND TESTING
x_train,x_test,y_train,y_test = train_test_split(features ,target, test_size= 0.2)
# MODEL CREATION
churn_model= LinearRegression()
churn_model.fit(x_train,y_train)
predict = churn_model.predict(features)
print("the predicted values are\n",predict)
# USING THE STOCHATIC GRADIENT DESCENT TO CONTINUALLY UPDATE THE DATA
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
sgd_model = SGDRegressor(max_iter=1000, eta0= 0.01, random_state= 42)
sgd_model.fit(features_scaled,target)
predictions = sgd_model.predict(features_scaled)
print("SGD PREDICTIONS ARE", predictions)
# ENSEMBLE METHODS USING THE BAGGING TECHNIQUES WHERE WE USE RANDOM TREE
from sklearn.ensemble import RandomForestClassifier

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# Evaluate
accuracy = rf_model.score(X_test, y_test)
print(f"The Accuracy of the model is: {accuracy}")