import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # encoding text to numbers

def	convert_dataset(df):
	# df = df.loc[df['CurrencySymbol'] == 'USD'] # only locate where CurrencySymbol = USD
	# df = df.drop(labels = 'CurrencySymbol', axis = 1) # drop means to remove label axis=1 means column

	df = df.loc[df['ConvertedSalary'] < 300000] # Only select convertedsalary of less than 250,000
	df = df.dropna() # drop all NAN rows
	# df = df.replace(',','', regex=True) # remove all commas from all elements
	# df = df.apply(pd.to_numeric, errors='ignore') # convert everything to int or floats?
	return df

# replace strings with numbers for analysis
def replace_yearscodingprof(df):
	df = df.replace(to_replace = ['0-2 years'], value = 1)
	df = df.replace(to_replace = ['3-5 years'], value = 4)
	df = df.replace(to_replace = ['6-8 years'], value = 7)
	df = df.replace(to_replace = ['9-11 years'], value = 10)
	df = df.replace(to_replace = ['12-14 years'], value = 13)
	df = df.replace(to_replace = ['15-17 years'], value = 16)
	df = df.replace(to_replace = ['18-20 years'], value = 19)
	df = df.replace(to_replace = ['21-23 years'], value = 22)
	df = df.replace(to_replace = ['24-26 years'], value = 25)
	df = df.replace(to_replace = ['27-29 years'], value = 28)
	df = df.replace(to_replace = ['30 or more years'], value = 30)
	return df

def	oaxaca():

	return 0



CSV_FILE = '../kfu/megainc.csv'

# load dataset
df = pd.read_csv(CSV_FILE)

seed = int(time.time())
# df = df.loc[0:75000].sample(2500, random_state=seed) # take random sample
df = df[['Gender', 'ConvertedSalary', 'YearsCodingProf']]


mdf = df.loc[df['Gender'] == 'Male']
# fdf = df.loc[df['Gender'] == 'Female']
# df = df.loc[df['Gender'].isin(['Male', 'Female'])] # Comparing multiple types using isin


mdf = convert_dataset(mdf)
mdf = replace_yearscodingprof(mdf)
mdf = convert_dataset(mdf)
print(mdf)


mdf.plot(kind='scatter', x = 'YearsCodingProf', y = 'ConvertedSalary')
plt.show()
