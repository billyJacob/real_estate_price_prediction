import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error

# this section of the code is just to view the records in the csv_file.
data = pd.read_csv("Real estate.csv")

# the describe function is used to get the full description of the dataset.
description = data.describe()
print(description)


# the corr method is used to get the correlation between the different attributes.
# this step is to find the highle correlated attributes so that they can be excluded from the dataset.
correlations = data.corr()
print(correlations)

# next we need to remove the latitude and longitude columns as they dont make any direct significant impact on the house price
data = data.drop(labels=["X5 latitude","X6 longitude", "No"], axis=1)

# the describe function is used to get the full description of the dataset.
correlations = data.corr()
print(correlations)


# the first four columns 0 - 3 are the features for predicting the house price
features = data.iloc[:,[0,3]].values
print(features)


label = data.iloc[:,4].values

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.2, random_state = 0)
#X_train= X_train.reshape(-1, 1)
#y_train= y_train.reshape(-1, 1)
#X_test = X_test.reshape(-1, 1)


regr = linear_model.LinearRegression()

regr.fit(X_train,y_train)


y_pred = regr.predict(X_test)

print(y_pred)