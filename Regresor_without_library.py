import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pandas as pd
from sklearn.metrics import r2_score
import timeit
import time


class MultipleLinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # Initialize weights and bias to zeros
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent
        for i in range(self.num_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

def train(X_train, y_train):
    regressor = MultipleLinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def main():
    dataset = pd.read_csv('x_train.csv')
    dataset1 = pd.read_csv('y_train.csv')
    dataset2 = pd.read_csv('x_test.csv')
    dataset3 = pd.read_csv('y_test.csv')

    X_train = dataset.iloc[:, :].values
    y_train = dataset1.iloc[:, :].values.flatten()
    X_test = dataset2.iloc[:, :].values
    y_test = dataset3.iloc[:, :].values.flatten()

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    start_time = time.time()
    regressor = train(X_train, y_train)
    elapsed_time = time.time() - start_time

    y_pred = regressor.predict(X_test)

    np.set_printoptions(precision=2)
    print("Values Predicted vs real data")
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
    print("\n")
    print(f"The R2 is: {r2_score(y_test, y_pred)}")
    print("\n")
    print(f"Elapsed time: {elapsed_time} seconds")
    print("\n")
    execution_time = timeit.timeit(lambda: train(X_train, y_train), number=100)
    # Print the execution time
    print("Execution time: {:.6f} seconds".format(execution_time))

main()
