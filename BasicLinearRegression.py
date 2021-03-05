#Python Data Analysis Library, for reading data from csv
import pandas as pd
#Tools for converting data to multidimensional arrays 
import numpy as np
#Sci-kit learn tool for running linear regression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
#Library for visualising the data
import matplotlib.pyplot as plt
#
import statsmodels.formula.api as sm


# load data set
data = pd.read_csv('./50_Startups.csv')

# values converts it into a numpy array: -1 means auto interpret number of rows, 1 means me want 1 column
X = data.iloc[:, :-1].values #.reshape(-1, 1)  
Y = data.iloc[:, 4].values #.reshape(-1, 1)

# linear_regressor = LinearRegression()  # create object for the class
# linear_regressor.fit(X, Y)  # perform linear regression
# Y_pred = linear_regressor.predict(X)  # make predictions

# data.plot(kind='scatter', x='R&D Spend', y='Profit')
# plt.plot(X, Y_pred, color='red')
# plt.show()

labelencoder = LabelEncoder()
X[:,3] = labelenencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

X = np.append(arr = np.ones((30,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2,3,4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


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


# data.plot(kind='scatter', x='R&D Spend', y='Profit')
# plt.plot(X_test, y_pred, color='red')
# plt.show()