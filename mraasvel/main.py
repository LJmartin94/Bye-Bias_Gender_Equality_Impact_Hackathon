import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # encoding text to numbers
import random
import time

def	convert_dataset(df):
	df = df.loc[df['ConvertedSalary'] < 250000] # Only select convertedsalary of less than 250,000
	df = df.loc[df['CurrencySymbol'] == 'USD'] # only locate where CurrencySymbol = USD
	df = df.drop(labels = 'CurrencySymbol', axis = 1) # drop means to remove label axis=1 means column

	df = df.dropna() # drop all NAN rows
	# df = df.replace(',','', regex=True) # remove all commas from all elements
	# df = df.apply(pd.to_numeric, errors='ignore') # convert everything to int or floats?
	return df




# load dataset
df = pd.read_csv('survey/data.csv')

seed = int(time.time())
df = df.loc[0:75000].sample(2500, random_state=seed)
df = df.iloc[:, [56,26,27]]

mdf = df.loc[df['Gender'] == 'Male']
fdf = df.loc[df['Gender'] != 'Male']
# df = df.loc[df['Gender'].isin(['Male', 'Female'])] # Comparing multiple types using isin

mdf = convert_dataset(mdf)
fdf = convert_dataset(fdf)




fdf.plot(kind='scatter', x = 'Gender', y = 'ConvertedSalary')
plt.show()
