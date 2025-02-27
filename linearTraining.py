import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

np.random.seed(0)
m =100
X = np.linspace(0, 10, m).reshape(m, 1)
y = X**2 + np.random.randn(m, 1)
  

# Load the data

model = SVR(C=1000)
model.fit(X, y)
prediction = model.predict(X)

# Plot the data

plt.scatter(X, y)
plt.plot(X, prediction, color='red')


plt.show()
