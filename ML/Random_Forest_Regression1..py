import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

x= x.reshape(-1, 1)

from sklearn.model_selection import train_test_split

x_train, y_train, x_text, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Fitting Random Forest Regression to the dataset
from  sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(x, y)

#y_pred = regressor.predict(6.5)

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color ='blue')
plt.title('Truth or Bluff(Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()