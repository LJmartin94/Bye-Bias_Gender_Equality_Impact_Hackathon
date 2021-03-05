import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


# load dataset

data = pd.read_csv('./test.csv')
print(data)

# data.corr() # correlation coefficient

data.plot(kind='scatter', x = 'Age', y = 'Salary')
data.plot(kind='line', x = 'Age', y = 'Salary')
plt.show()

Age = pd.DataFrame(data['Age'])
Salary = pd.DataFrame(data['Salary'])
# print(Age)
# print(Salary)


# build linear regression model

lm = linear_model.LinearRegression()
model = lm.fit(Age, Salary)

# print(model.coef_)
# print(model.intercept_)
# print(model.score(Age, Salary)) # evaluate the model

# predict new value of Salary
age_new = [[30], [35], [22]]
salary_prediction = model.predict(age_new)
print(salary_prediction)
