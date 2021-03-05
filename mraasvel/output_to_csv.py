import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # encoding text to numbers
import random


# load dataset
dataset = pd.read_csv('survey/data.csv')

dataset_sub = dataset.loc[0:75000].sample(10, random_state=42) # random state (0:75000 is range) (10 = amount, random_state is seed)
dataset_sub = dataset_sub.iloc[:,[26, 27]] # give column numbers that you want to extract (0-61)


print(dataset_sub)
dataset_sub.to_csv("./out.csv")


