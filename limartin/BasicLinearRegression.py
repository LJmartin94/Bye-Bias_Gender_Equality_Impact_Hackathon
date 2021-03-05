#Python Data Analysis Library, for reading data from csv
import pandas as pd
#Tools for converting data to multidimensional arrays 
import numpy as np
#Sci-kit learn tool for running linear regression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
#Library for visualising the data
import matplotlib.pyplot as plt


# load data set
#data = pd.read_csv('./50_Startups.csv')
df = pd.read_csv('../megainc.csv')

df = df.loc[df['ConvertedSalary'] < 300000] # Only select convertedsalary of less than 250,000
df = df.dropna()

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

df = replace_yearscodingprof(df)

df1 = df
# values converts it into a numpy array: -1 means auto interpret number of rows, 1 means me want 1 column
X = df1.loc[:, 'YearsCodingProf'].values.reshape(-1, 1)  
Y = df1.loc[:, 'ConvertedSalary'].values.reshape(-1, 1)

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions

df1.plot(kind='scatter', x='YearsCodingProf', y='ConvertedSalary')
plt.plot(X, Y_pred, color='red')
plt.show()
print("Coefficient - Extra salary on average per additional year of coding professionally (globally):")
print(linear_regressor.coef_)


df2 = df
df2 = df.loc[df['Gender'] == 'Male']
# values converts it into a numpy array: -1 means auto interpret number of rows, 1 means me want 1 column
X = df2.loc[:, 'YearsCodingProf'].values.reshape(-1, 1)  
Y = df2.loc[:, 'ConvertedSalary'].values.reshape(-1, 1)

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions

df2.plot(kind='scatter', x='YearsCodingProf', y='ConvertedSalary')
plt.plot(X, Y_pred, color='red')
plt.show()
print("Coefficient - Extra salary on average per additional year of coding professionally (Male):")
print(linear_regressor.coef_)


df3 = df
df3 = df.loc[df['Gender'] != 'Male']
# values converts it into a numpy array: -1 means auto interpret number of rows, 1 means me want 1 column
X = df3.loc[:, 'YearsCodingProf'].values.reshape(-1, 1)  
Y = df3.loc[:, 'ConvertedSalary'].values.reshape(-1, 1)

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions

df3.plot(kind='scatter', x='YearsCodingProf', y='ConvertedSalary')
plt.plot(X, Y_pred, color='red')
plt.show()
print("Coefficient - Extra salary on average per additional year of coding professionally (non Male identifying):")
print(linear_regressor.coef_)





# #Convert the column into categorical columns
# states=pd.get_dummies(X['State'],drop_first=True)
# # Drop the state coulmn
# X=X.drop('State',axis=1)
# # concat the dummy variables
# X=pd.concat([X,states],axis=1)
# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
# # Fitting Multiple Linear Regression to the Training set
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# # Predicting the Test set results
# y_pred = regressor.predict(X_test)
# from sklearn.metrics import r2_score
# score=r2_score(y_test,y_pred)


# print(regr.coef_)





# labelencoder = LabelEncoder()
# X[:,3] = labelenencoder.fit_transform(X[:,3])
# onehotencoder = OneHotEncoder(categorical_features = [3])
# X = onehotencoder.fit_transform(X).toarray()

# X = X[:, 1:]

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)


##Dropping variables based on p value
# # X = np.append(arr = np.ones((30,1)).astype(int), values = X, axis = 1)
# # X_opt = X[:,[0,1,2,3,4,5]]
# # regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# # regressor_OLS.summary()

# # X_opt = X[:,[0,1,2,3,4]]
# # regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# # regressor_OLS.summary()

# # X_opt = X[:,[0,1,2,3]]
# # regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# # regressor_OLS.summary()






# data.plot(kind='scatter', x='R&D Spend', y='Profit')
# plt.plot(X_test, y_pred, color='red')
# plt.show()