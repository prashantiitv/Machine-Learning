import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#reading data
dataframe = pd.read_fwf('dataset.txt')
X = dataframe[['Brain']]
Y = dataframe[['Body']]

#training model
body_reg = linear_model.LinearRegression()
body_reg.fit(X, Y)

#visulaising results
plt.scatter(X, Y)
plt.plot(X, body_reg.predict(X))
plt.show()
