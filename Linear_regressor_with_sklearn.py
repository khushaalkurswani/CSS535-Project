# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import timeit
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initializing an empty list to store the elapsed times
elapsed_times = []

# Defining the training function to train a linear regression model using the given training data
def train(X_train, y_train):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor

# Defining the main function that loads the training and testing data, trains the model, and evaluates its performance
def main():
    # Loading the training and testing data from CSV files
    dataset = pd.read_csv('x_train.csv')
    dataset1 = pd.read_csv('y_train.csv')
    dataset2 = pd.read_csv('x_test.csv')
    dataset3 = pd.read_csv('y_test.csv')

    # Defining the number of samples, features, and cores
    n=3633 #Number of samples
    m=4   #Number of features
    p=2   #Number of cores

    # Splitting the training and testing data into feature and label sets
    X_train = dataset.iloc[:n, :].values
    y_train = dataset1.iloc[:n, :].values
    X_test = dataset2.iloc[:, :].values
    y_test = dataset3.iloc[:, :].values

    # Scaling the features using the MinMaxScaler from Scikit-learn
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Training the model on the scaled training data
    regressor = train(X_train, y_train)

    # Predicting the labels for the testing data using the trained model
    y_pred = regressor.predict(X_test)

    # Calculating the average training time for the model using timeit library
    mean_time = timeit.Timer(lambda: [train(X_train, y_train) for _ in range(100)]).timeit(1) / 100

    # Setting the printing options for numpy arrays
    np.set_printoptions(precision=2)

    # Printing the predicted values vs real data for the testing data
    print("Values Predicted vs real data")
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
    print("\n")

    # Calculating and printing the R2 score for the model's performance evaluation
    print(f"The R2 is: {r2_score(y_test, y_pred)}")
    print("\n")

    # Printing the average elapsed time for the model training
    print(f"The Average elapsed time: {mean_time} seconds")
    print("\n")

    # Calculating and printing the execution rate (FLOPS) of the model
    FLOPS=(2 * n * m^2 * p) / mean_time
    print(f"The execution rate(FLOPS) is: {FLOPS}")

# Calling the main function to run the code
main()
        
        
        