import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import timeit
import time
from sklearn.preprocessing import StandardScaler


def train(X_train, y_train):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def main():
    dataset = pd.read_csv('x_train.csv')
    dataset1 = pd.read_csv('y_train.csv')
    dataset2 = pd.read_csv('x_test.csv')
    dataset3 = pd.read_csv('y_test.csv')

    X_train = dataset.iloc[:, :].values
    y_train = dataset1.iloc[:, :].values
    X_test = dataset2.iloc[:, :].values
    y_test = dataset3.iloc[:, :].values

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print(X_train)

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
    execution_time = timeit.timeit(lambda: train(X_train, y_train), number=10000)
    # Print the execution time
    print("Execution time: {:.6f} seconds".format(execution_time))
main()
