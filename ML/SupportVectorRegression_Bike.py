import  numpy as np
import  pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('bike_sharing.csv')

x = dataset.iloc[:, -7].values
y = dataset.iloc[:, -1].values

x = x.reshape(-1, 1)
#y = y.reshape(-1, 1)
print (x)
print (y)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x =  sc_x.fit_transform(x)
y =  sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x, y)

#y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff(SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



