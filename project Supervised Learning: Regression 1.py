
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.datasets import load_boston

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

plt.plot(y,X)
plt.show()
regr = linear_model.LinearRegression()
regr.fit(X,y)
y_predict = regr.predict(X)
plt.plot(y_predict,X)
plt.show()
print(regr.coef_)
print(regr.intercept_)
X_future = np.array(range(1, 11))
X_future = X_future.reshape(-1, 1)
future_predict = regr.predict(X_future)
plt.plot(X_future,future_predict)
plt.show()

    