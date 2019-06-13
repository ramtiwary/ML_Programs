import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as sn
from sklearn import model_selection
data = pd.read_csv('bike_sharing.csv')
print(data.shape)
print (data.head())

data.rename(columns={'weathersit':'weather',
                     'mnth':'month',
                     'hr':'hour',
                     'hum': 'humidity',
                     'cnt':'count'},inplace=True)

data = data.drop(['instant','dteday','yr'], axis=1)

data['season'] = data.season.astype('category')
data['month'] = data.month.astype('category')
data['hour'] = data.hour.astype('category')
data['holiday'] = data.holiday.astype('category')
data['weekday'] = data.weekday.astype('category')
data['workingday'] = data.workingday.astype('category')
data['weather'] = data.weather.astype('category')

print(data.dtypes)

fig, ax = plt.subplots(figsize=(20,10))
sn.pointplot(data=data[['hour',
                       'count',
                       'weekday']],
            x='hour', y='count',
            hue='weekday', ax=ax)
print(ax.set(title="Use of the system during weekdays and weekends"))
plt.show()

fig, ax = plt.subplots(figsize=(20,10))
sn.pointplot(data=data[['hour',
                       'casual',
                       'weekday']],
            x='hour', y='casual',
            hue='weekday', ax=ax)
print(ax.set(title="Use of the system by casual Uesrs"))
plt.show()

fig, ax = plt.subplots(figsize=(20,10))
sn.pointplot(data=data[['hour',
                       'registered',
                       'weekday']],
            x='hour', y='registered',
            hue='weekday', ax=ax)
print(ax.set(title="Use of the system by registered Uesrs"))
plt.show()

fig, ax = plt.subplots(figsize=(20,10))
sn.pointplot(data=data[['hour',
                       'registered',
                       'weekday']],
            x='hour', y='registered',
            hue='weekday', ax=ax)
print(ax.set(title="Use of the system by casual Uesrs"))
plt.show()
