import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

#Convert the columns into Categarical cloumn
states = pd.get_dummies(x['State'],drop_first = True)

x = x.drop('State', axis=1)

x = pd.concat([x, states], axis=1)

#Splitting the dataset into the training set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

#Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

viz_train = plt
viz_train.plot(x_train, y_train, color='red')
viz_train.plot(x_train, reg.predict(x_train), color='blue')
r2 = reg.score(x_train, reg.predict(x_train))
viz_train.title('Salary VS Experience (Training set)')
viz_train.xlabel('Year of Experience')
viz_train.ylabel('Salary')
viz_train.show()

y_pred = reg.predict(x_test)
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print (score)

