from collections import Counter
from matplotlib.colors import ListedColormap
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    def fit(self, X, y):
        #intialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        #gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X,self.weights)+self.bias #wX+b
            dw = (1/n_samples) * np.dot(X.T,(y_predicted - y)) #derivative w.r.t weights
            db = (1/n_samples) * np.sum(2*(y_predicted-y)) #derivative w.r.t bias

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        y_predicted = np.dot(X,self.weights)+self.bias
        return y_predicted

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


X,y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234) 

# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:,0],y,color='b',marker='o',s=30)
# plt.show()

# print (X_train.shape)


linear = LinearRegression(learning_rate=0.0001, n_iters=100000)
linear.fit(X_train, y_train)
predictions = linear.predict(X_test)
print(mean_squared_error(y_test, predictions))

plt.scatter(y_test, predictions)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual Values (y_test)")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values - Linear Regression")
plt.show()