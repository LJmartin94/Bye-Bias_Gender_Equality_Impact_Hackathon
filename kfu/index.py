# **************************************************************************** #
#                                                                              #
#                                                         ::::::::             #
#    index.py                                           :+:    :+:             #
#                                                      +:+                     #
#    By: kfu <kfu@student.codam.nl>                   +#+                      #
#                                                    +#+                       #
#    Created: 2021/03/04 14:25:22 by kfu           #+#    #+#                  #
#    Updated: 2021/03/05 11:40:51 by kfu           ########   odam.nl          #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


data = pd.read_csv("megainc.csv")
data.drop(['Respondent'], axis = 1, inplace=True)
data.dropna(axis=0, how='all', thresh=58)

print(data)
