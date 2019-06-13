#1.machine learning model to predict salary based on experience for
#a given dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Visualizing the Training set results
viz_train = plt
viz_train.scatter(X_train, Y_train, color='red')
viz_train.plot(X_train, regressor.predict(X_train), color='blue')
r2 = regressor.score(X_train, regressor.predict(X_train))
viz_train.title('Salary VS Experience (Training set)')
viz_train.xlabel('Year of Experience')
viz_train.ylabel('Salary')
#viz_train.show()
print(r2)

# Visualizing the Test set results
viz_test = plt
viz_test.scatter(X_test, Y_test, color='red')
viz_test.plot(X_train, regressor.predict(X_train), color='blue')
viz_test.title('Salary VS Experience (Test set)')
viz_test.xlabel('Year of Experience')
viz_test.ylabel('Salary')
viz_test.show()

# Predicting the result of 5 Years Experience
#y_pred = regressor.predict(5)

#y_pred = regressor.predict(X_test)



