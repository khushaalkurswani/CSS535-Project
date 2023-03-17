#Regression from scratch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pandas as pd
from sklearn.metrics import r2_score
import timeit
import time
from sklearn.preprocessing import MinMaxScaler


# Define a class for multiple linear regression
class MultipleLinearRegression:
    # Constructor method that initializes the object with default values for learning rate and number of iterations
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        # Assign the given learning rate and number of iterations to object variables
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        # Set the initial weights and bias to None
        self.weights = None
        self.bias = None

    # Method to fit the regression model using the given input data X and output data y
    def fit(self, X, y):
        # Get the number of samples and features in the input data X
        num_samples, num_features = X.shape

        # Initialize weights and bias to zeros
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Perform gradient descent for the specified number of iterations
        for i in range(self.num_iterations):
            # Calculate the predicted output y for the given input X
            y_predicted = np.dot(X, self.weights) + self.bias
            # Calculate the gradients of the cost function with respect to the weights and bias
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            # Update the weights and bias using the calculated gradients and the learning rate
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    # Method to predict the output y for the given input data X using the trained weights and bias
    def predict(self, X):
        # Calculate the predicted output y for the given input X using the trained weights and bias
        y_predicted = np.dot(X, self.weights) + self.bias
        # Return the predicted output y
        return y_predicted

def train(X_train, y_train):
    regressor = MultipleLinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def main():
    #Reading the dataset
    dataset = pd.read_csv('x_train.csv')
    dataset1 = pd.read_csv('y_train.csv')
    dataset2 = pd.read_csv('x_test.csv')
    dataset3 = pd.read_csv('y_test.csv')

    n = 3633  # Number of samples
    m = 4  # Number of features
    p = 2  # Number of cores
    #flattening the dataset
    X_train = dataset.iloc[:n, :].values
    y_train = dataset1.iloc[:n, :].values.flatten()
    X_test = dataset2.iloc[:, :].values
    y_test = dataset3.iloc[:, :].values.flatten()
    #Normalization
    scaler = MinMaxScaler()
    X_train= scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)

    print(X_train)
    #Apply regression
    regressor = train(X_train, y_train)
    #Predict values
    y_pred = regressor.predict(X_test)
    print("Predicted values: ")
    print(y_pred)

    mean_time = timeit.Timer(lambda: [train(X_train, y_train) for _ in range(100)]).timeit(1) / 100

    np.set_printoptions(precision=2)
    print("Values Predicted vs real data")
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
    print("\n")
    print(f"The R2 is: {r2_score(y_test, y_pred)}")
    print("\n")
    print(f"The Average elapsed time: {mean_time} seconds")
    print("\n")
    # To calculate the execution rate we used: FLOPS = (2 * n * m^2 * p) / t
    FLOPS = (2 * n * m ^ 2 * p) / mean_time
    print(f"The execution rate(FLOPS) is: {FLOPS}")

main()