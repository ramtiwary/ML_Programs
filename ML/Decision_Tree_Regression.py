import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
#%matplotlib inline

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.cross_validation import train_test_spilit
x_train, y_train, x_test, y_test = train_test_spilit(x, y, test_size = 0.2, random_state=0)

y_pred = regressor.predict(6.5)

#Fitting the Decision Tree Regression to the Dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

#y_pred = regressor.predict(6.5)

plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

'''print(dataset.head())

x = dataset['Level'].values
y = dataset['Salary'].values
x = x.reshape(-1,1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(x_train, y_train)

y_pred =reg.predict(x_test)
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(df)

viz_train = plt
viz_train.scatter(x_train, y_train, color='red')
viz_train.plot(x_train, reg.predict(x_train), color='blue')
#r2 = reg.score(x_train, reg.predict(x_train))
viz_train.title('Salary VS Designation (Training set)')
#viz_train.xlabel('Yea')
#viz_train.ylabel('Salary')
viz_train.show()'''

