import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time

#Sci-kit learn tool for running linear regression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

def random_sample(df):
	seed = int(time.time())
	df = df.loc[0:75000].sample(1000, random_state=seed) # take random sample
	return df

# replace strings with numbers for analysis (yearscoding only)
def replace_yearscodingprof(df):
	df['YearsCodingProf'] = df['YearsCodingProf'].replace(to_replace = ['0-2 years'], value = 1)
	df['YearsCodingProf'] = df['YearsCodingProf'].replace(to_replace = ['3-5 years'], value = 4)
	df['YearsCodingProf'] = df['YearsCodingProf'].replace(to_replace = ['6-8 years'], value = 7)
	df['YearsCodingProf'] = df['YearsCodingProf'].replace(to_replace = ['9-11 years'], value = 10)
	df['YearsCodingProf'] = df['YearsCodingProf'].replace(to_replace = ['12-14 years'], value = 13)
	df['YearsCodingProf'] = df['YearsCodingProf'].replace(to_replace = ['15-17 years'], value = 16)
	df['YearsCodingProf'] = df['YearsCodingProf'].replace(to_replace = ['18-20 years'], value = 19)
	df['YearsCodingProf'] = df['YearsCodingProf'].replace(to_replace = ['21-23 years'], value = 22)
	df['YearsCodingProf'] = df['YearsCodingProf'].replace(to_replace = ['24-26 years'], value = 25)
	df['YearsCodingProf'] = df['YearsCodingProf'].replace(to_replace = ['27-29 years'], value = 28)
	df['YearsCodingProf'] = df['YearsCodingProf'].replace(to_replace = ['30 or more years'], value = 30)
	return df

def replace_ages(df):
	return df

def	convert_dataset(df):

	df = df.loc[df['ConvertedSalary'] < 300000] # Only select convertedsalary of less than 250,000
	df = df.dropna() # drop all NAN rows
	df = replace_yearscodingprof(df)
	# df = df.replace(',','', regex=True) # remove all commas from all elements
	# df = df.apply(pd.to_numeric, errors='ignore') # convert everything to int or floats?
	return df

def	oaxaca():
	return 0


# Calculate linear regression coefficients and intercept
def linear_coefficients(df, group):
	X = df.loc[:, 'YearsCodingProf'].values.reshape(-1, 1)
	Y = df.loc[:, 'ConvertedSalary'].values.reshape(-1, 1)

	linear_regressor = LinearRegression()  # create object for the class
	linear_regressor.fit(X, Y)  # perform linear regression
	Y_pred = linear_regressor.predict(X)  # make predictions

	df.plot(kind='scatter', x='YearsCodingProf', y='ConvertedSalary')
	plt.plot(X, Y_pred, color='red')

	# printing coefficients
	print("GROUP: ", group)
	print("Coefficient: ", linear_regressor.coef_)
	print("Intercept: ", linear_regressor.intercept_)

	file_name = './graphs/' + group + '.png'
	plt.savefig(file_name)
	return


# Function to print entry count
def print_entries(df, group):
	print(group, ": ", len(df))
	return

def print_counts(df):
	ndf = df['Gender']
	ndf = ndf.dropna(how = 'Gender')

	mdf = ndf.loc[df['Gender'] == 'Male']
	print_entries(mdf, 'Male')

	mdf = ndf.loc[df['Gender'] == 'Female']
	print_entries(mdf, 'Non-Male')

	mdf = ndf.loc[~df['Gender'].isin(['Male', 'Female'])]
	print_entries(mdf, 'Non Male and Non female')
	return

def get_kinship(df):
	df = df[['Gender', '421', 'Age']]
	df = df.dropna()
	df['421'] = df['421'].astype(int)

	mdf = df.loc[df['Gender'] == 'Male']
	# fdf = df.loc[df['Gender'] == 'Female']
	# ndf = df.loc[~df['Gender'].isin(['Female', 'Male'])]


	print(mdf['421'])
	mdf.plot.bar(x = '421')
	plt.show()


	# plt.savefig("./graphs/male_ad1.png")
	# fdf.421.value_counts().reindex(["Under 18 years old", "18 - 24 years old", "25 - 34 years old", "35 - 44 years old", "45 - 54 years old", "55 - 64 years old", "65 years or older"]).plot(kind="bar")
	# plt.savefig("./graphs/female_ad1.png")
	# ndf.421.value_counts().reindex(["Under 18 years old", "18 - 24 years old", "25 - 34 years old", "35 - 44 years old", "45 - 54 years old", "55 - 64 years old", "65 years or older"]).plot(kind="bar")
	# plt.savefig("./graphs/non_mf_ad1.png")

	plt.show()
	return


CSV_FILE = '../megainc.csv'
# CSV_FILE = '../kfu/megainc.csv'
groupm = 'Male'
groupf = 'Female'
group3 = 'Not Male'

# load dataset
df = pd.read_csv(CSV_FILE)

get_kinship(df)
exit()

# print_counts(df)

# Randomized samples
# df = random_sample(df)

edf = df[['Age', 'Gender']]
edf = edf.dropna()


# max width
# pd.set_option('display.max_colwidth', None)

# mdf = edf[edf['Gender'].isin(['Male', 'Female'])]
# mdf = edf[edf['Gender'] == 'Male']
# fdf = edf[edf['Gender'] == 'Female']

# mdf.Age.value_counts().reindex(["Under 18 years old", "18 - 24 years old", "25 - 34 years old", "35 - 44 years old", "45 - 54 years old", "55 - 64 years old", "65 years or older"]).plot(kind="bar")
# plt.savefig("./graphs/male_age.png")
# fdf.Age.value_counts().reindex(["Under 18 years old", "18 - 24 years old", "25 - 34 years old", "35 - 44 years old", "45 - 54 years old", "55 - 64 years old", "65 years or older"]).plot(kind="bar")
# plt.savefig("./graphs/female_age.png")

df = df[['Gender', 'ConvertedSalary', 'YearsCodingProf', 'Age']]


mdf = df.loc[df['Gender'] == 'Male']
fdf = df.loc[df['Gender'] == 'Female']
ndf = df.loc[df['Gender'] != 'Male']
# df = df.loc[df['Gender'].isin(['Male', 'Female'])] # Comparing multiple types using isin


mdf = convert_dataset(mdf)
fdf = convert_dataset(fdf)
ndf = convert_dataset(ndf)



linear_coefficients(mdf, groupm)
linear_coefficients(fdf, groupf)
linear_coefficients(ndf, group3)


