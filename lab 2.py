import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing._label import LabelEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# TASK ONE(1)
# MANIPULATION OF DATA SET USING THE PANDAS LIBARY AND CARRYING OUT THE PROCESS OF PRE-PROCESSING

df = pd.read_csv("data set\lab 2.csv")
print(df.shape)
print(df.describe())
# NULL CHECK
print(df.isnull().sum())
# DUPLICATE CHECK
print(df.duplicated())
print(df.dtypes)
df = df.drop('monthly_minutes', axis=1)
df = df.drop('region', axis =1)

print(df.dtypes)
print(df.head(3))
target = df['churn']
features= df.drop(columns = ['churn'])
# SPLITING THE DATA SET INTO TRAINING AND TESTING
x_train,x_test,y_train,y_test = train_test_split(features ,target, test_size= 0.3)
# TASK TWO(2)
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
from  sklearn.metrics import accuracy_score
acc= churn_model.score(x_test,y_test)
print(acc)



