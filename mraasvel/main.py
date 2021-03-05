import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time

#Sci-kit learn tool for running linear regression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

def random_sample(df, total):
	seed = int(time.time())
	df = df.loc[0:75000].sample(total, random_state=seed) # take random sample
	return df

# Disgusting code, don't look if you want keep your eyes

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
	df['Age'] = df['Age'].replace(to_replace = ['Under 18'], value = 18)
	df['Age'] = df['Age'].replace(to_replace = ['18 - 24 years old'], value = 20)
	df['Age'] = df['Age'].replace(to_replace = ['25 - 34 years old'], value = 30)
	df['Age'] = df['Age'].replace(to_replace = ['35 - 44 years old'], value = 40)
	df['Age'] = df['Age'].replace(to_replace = ['45 - 54 years old'], value = 50)
	df['Age'] = df['Age'].replace(to_replace = ['55 - 64 years old'], value = 59)
	df['Age'] = df['Age'].replace(to_replace = ['65 years or older'], value = 65)
	return df

def replace_bootcamp(df):
	df['TimeAfterBootcamp'] = df['TimeAfterBootcamp'].replace(to_replace = ['I already had a full-time job as a developer when I began the program'], value = 'Before')
	df['TimeAfterBootcamp'] = df['TimeAfterBootcamp'].replace(to_replace = ['Immediately after graduating'], value = '0')
	df['TimeAfterBootcamp'] = df['TimeAfterBootcamp'].replace(to_replace = ['Less than a month'], value = '< 1')
	df['TimeAfterBootcamp'] = df['TimeAfterBootcamp'].replace(to_replace = ['One to three months'], value = '1-3')
	df['TimeAfterBootcamp'] = df['TimeAfterBootcamp'].replace(to_replace = ['Four to six months'], value = '4-6')
	df['TimeAfterBootcamp'] = df['TimeAfterBootcamp'].replace(to_replace = ['Six months to a year'], value = '6-12')
	df['TimeAfterBootcamp'] = df['TimeAfterBootcamp'].replace(to_replace = ['Longer than a year'], value = '12+')
	df['TimeAfterBootcamp'] = df['TimeAfterBootcamp'].replace(to_replace = ['I haven’t gotten a developer job'], value = 'Never')
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
	print("Intercept: ", linear_regressor.intercept_, "\n")

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
	df = df[['Gender', '421', 'YearsCodingProf']]
	df = df.dropna()
	df = replace_yearscodingprof(df)
	df['421'] = df['421'].astype(int)

	# means hoyp
	for i in [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 30]:
		mdf = df.loc[df['Gender'] == 'Male']
		fdf = df.loc[df['Gender'] == 'Female']
		mdf = mdf.loc[df['YearsCodingProf'] == i]
		fdf = fdf.loc[df['YearsCodingProf'] == i]

		meanm = mdf['421'].mean()
		meanf = fdf['421'].mean()
		print("Male average score, with ", i, "years of coding exp: ", meanm)
		print("Female average score, with", i, "years of coding exp: ", meanf)
	return

def bootcamp_hyp1(df):
	df = df[['Gender', 'TimeAfterBootcamp']]

	df = df.dropna()

	df = replace_bootcamp(df)

	mdf = df.loc[df['Gender'] == 'Male']
	fdf = df.loc[df['Gender'] == 'Female']
	ndf = df.loc[~df['Gender'].isin(['Male', 'Female'])]

	print(ndf)
	mdf.TimeAfterBootcamp.value_counts().reindex(["Before", "0", "< 1", "1-3", "4-6", "6-12", "12+", "Never"]).plot(kind="bar")
	plt.savefig("./graphs/male_bootcamp.png")
	fdf.TimeAfterBootcamp.value_counts().reindex(["Before", "0", "< 1", "1-3", "4-6", "6-12", "12+", "Never"]).plot(kind="bar")
	plt.savefig("./graphs/female_bootcamp.png")
	ndf.TimeAfterBootcamp.value_counts().reindex(["Before", "0", "< 1", "1-3", "4-6", "6-12", "12+", "Never"]).plot(kind="bar")
	plt.savefig("./graphs/none_mf_bootcamp.png")
	return

def parental_leave(df):
	return

# CSV_FILE = '../megainc.csv'
CSV_FILE = '../kfu/megainc.csv'
groupm = 'Male'
groupf = 'Female'
group3 = 'Not Male or Female'

# load dataset
df = pd.read_csv(CSV_FILE)

parental_leave(df)
exit()

# Randomized samples
# df = random_sample(df, 2500)


# max width
# pd.set_option('display.max_colwidth', None)

# mdf = edf[edf['Gender'].isin(['Male', 'Female'])]
# mdf.Age.value_counts().reindex(["Under 18 years old", "18 - 24 years old", "25 - 34 years old", "35 - 44 years old", "45 - 54 years old", "55 - 64 years old", "65 years or older"]).plot(kind="bar")
# plt.savefig("./graphs/male_age.png")

df = df[['Gender', 'ConvertedSalary', 'YearsCodingProf', 'Age']]


mdf = df.loc[df['Gender'] == 'Male']
fdf = df.loc[df['Gender'] == 'Female']
ndf = df.loc[~df['Gender'].isin(['Male', 'Female'])]
# df = df.loc[df['Gender'].isin(['Male', 'Female'])] # Comparing multiple types using isin


mdf = convert_dataset(mdf)
fdf = convert_dataset(fdf)
ndf = convert_dataset(ndf)



linear_coefficients(mdf, groupm)
linear_coefficients(fdf, groupf)
linear_coefficients(ndf, group3)
