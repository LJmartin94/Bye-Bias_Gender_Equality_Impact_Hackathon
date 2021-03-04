#Python Data Analysis Library
import pandas as pd
#Library for visualising and plotting data in python
import matplotlib.pyplot as plt
#Sci-kit learn tool for plotting linear models
from sklearn import linear_model

data=pd.read_csv('/home/lindsay/ghBiasHackathon/wastedata.csv')

data
data.shape
data.plot(kind='scatter', x='waist', y='weight')
plt.show()

data.plot(kind='box')
plt.show()

data.corr() #Shows the correlation coeffiecient

