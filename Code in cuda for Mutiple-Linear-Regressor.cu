#include <cuda_runtime.h> // Include CUDA runtime API header files
#include <cuda.h>
#include "device_launch_parameters.h"
#include <math.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>

using namespace std;

__global__ void train(float *x, float *y, float *theta, float alpha, int m, int n) // Define kernel function 'train'
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Calculate thread ID based on block and thread index

    if (tid < n)
    {                    // Check if thread ID is within range
        float sum = 0.0; // Initialize sum to zero
        for (int i = 0; i < m; i++)
        {                                                                     // Loop through training examples
            sum += (theta[0] + theta[1] * x[i] + theta[2] * x[i + m]) - y[i]; // Compute sum of errors
        }
        if (tid == 0)
        {                                        // Check if thread is the first thread in the block
            theta[0] -= alpha * (1.0 / m) * sum; // Update theta[0]
        }
        else if (tid == 1)
        {                                          // Check if thread is the second thread in the block
            sum *= x[tid - 1];                     // Multiply sum by x[0]
            theta[tid] -= alpha * (1.0 / m) * sum; // Update theta[1]
        }
        else if (tid == 2)
        {                                          // Check if thread is the third thread in the block
            sum *= x[tid - 2 + m];                 // Multiply sum by x[1]
            theta[tid] -= alpha * (1.0 / m) * sum; // Update theta[2]
        }
    }
}

class Regressor
{ // Define Regressor class
private:
    float *theta, *d_x, *d_y, *d_theta; // Declare pointers for theta, x, y, and theta on device
    int m, n;                           // Declare number of training examples and features

public:
    Regressor(int m, int n) // Constructor
    {
        this->m = m;          // Set number of training examples
        this->n = n;          // Set number of features
        theta = new float[n]; // Allocate memory for theta
        for (int i = 0; i < n; i++)
        { // Initialize theta to zero
            theta[i] = 0.0;
        }
        cudaMalloc(&d_x, m * n * sizeof(float)); // Allocate memory for x on device
        cudaMalloc(&d_y, m * sizeof(float));     // Allocate memory for y on device
        cudaMalloc(&d_theta, n * sizeof(float)); // Allocate memory for theta on device
    }

    ~Regressor() // Destructor
    {
        delete[] theta;    // Free memory for theta
        cudaFree(d_x);     // Free memory for x on device
        cudaFree(d_y);     // Free memory for y on device
        cudaFree(d_theta); // Free memory for theta on device
    }

    void fit(float *x_train, float *y_train, float alpha, int iterations)
    {
        // Copy the training data and initial parameters from host to device
        cudaMemcpy(d_x, x_train, m * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y_train, m * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_theta, theta, n * sizeof(float), cudaMemcpyHostToDevice);

        // Determine the block and grid sizes for the kernel launch
        int block_size = 1024;
        int grid_size = (n + block_size - 1) / block_size;

        // Run the training loop for the specified number of iterations
        for (int i = 0; i < iterations; i++)
        {
            train<<<grid_size, block_size>>>(d_x, d_y, d_theta, alpha, m, n);
            // Wait for the kernel to finish executing before continuing
            cudaDeviceSynchronize();
        }

        // Copy the final parameters from device back to host
        cudaMemcpy(theta, d_theta, n * sizeof(float), cudaMemcpyDeviceToHost);
    }

    float predict(float *x_test)
    {
        float y_pred = 0.0;
        // Calculate the predicted value using the learned parameters
        for (int i = 0; i < n; i++)
        {
            y_pred += theta[i] * x_test[i];
        }
        // Return the predicted value
        return y_pred;
    }
};
float *parseCSV(string fName, int &m, int &n)
{
    ifstream data(fName);
    string line;
    string item;
    vector<vector<float>> dataMatrix;
    getline(data, line); // skip the title
    while (getline(data, line))
    {
        vector<float> dataVec;
        string data;
        stringstream lineStream(line);

        while (getline(lineStream, item, ','))
        {
            dataVec.push_back(stof(item));
        }
        dataMatrix.push_back(dataVec);
    }
    m = dataMatrix.size();
    n = dataMatrix.at(0).size();
    float *a = new float[m * n];

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            a[i * n + j] = dataMatrix[i][j];
        }
    }

    return a;
};
void printMatrix(float *A, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << A[i * n + j] << " ";
        }
        cout << endl;
    }
}
int main()
{
    int m, n, y_trainM, y_trainN, x_testM, x_testN;
    // parseCSV("x_test.csv", m, n);

    /*
    float* x_train = new float[m * n];
    float* y_train = new float[m];
    float* x_test = new float[n];

    // fill x_train and y_train with data from x_train.csv and y_train.csv
    // fill x_test with data from x_test.csv
*/

    // example training data
    float *x_train = parseCSV("x_train.csv", m, n);
    float *y_train = parseCSV("y_train.csv", y_trainM, y_trainN);

    // example test data
    float *x_test = parseCSV("x_test.csv", x_testM, x_testN);
    // copy data to device
    float *d_x_train, *d_y_train, *d_x_test, *d_y_pred;
    cudaMalloc(&d_x_train, m * n * sizeof(float));                                 // Allocate memory on device for x_train
    cudaMalloc(&d_y_train, m * sizeof(float));                                     // Allocate memory on device for y_train
    cudaMalloc(&d_x_test, n * sizeof(float));                                      // Allocate memory on device for x_test
    cudaMalloc(&d_y_pred, sizeof(float));                                          // Allocate memory on device for the predicted y value
    cudaMemcpy(d_x_train, x_train, m * n * sizeof(float), cudaMemcpyHostToDevice); // Copy x_train to device
    cudaMemcpy(d_y_train, y_train, m * sizeof(float), cudaMemcpyHostToDevice);     // Copy y_train to device
    cudaMemcpy(d_x_test, x_test, n * sizeof(float), cudaMemcpyHostToDevice);       // Copy x_test to device

    // train model
    Regressor regressor(m, n);                              // Create a new instance of the Regressor class with m and n
    float alpha = 0.01;                                     // Set the learning rate alpha
    int iterations = 1000;                                  // Set the number of training iterations
    regressor.fit(d_x_train, d_y_train, alpha, iterations); // Fit the model to the training data

    // test model
    float y_pred = regressor.predict(x_test);                             // Predict the output for the test data point
    cudaMemcpy(&y_pred, d_y_pred, sizeof(float), cudaMemcpyDeviceToHost); // Copy the predicted y value to the host
    cout << "Predicted value: " << y_pred << endl;                        // Print the predicted y value

    // free memory
    cudaFree(d_x_train); // Free the memory allocated for x_train on the device
    cudaFree(d_y_train); // Free the memory allocated for y_train on the device
    cudaFree(d_x_test);  // Free the memory allocated for x_test on the device
    cudaFree(d_y_pred);  // Free the memory allocated for the predicted y value on the device

    delete[] x_test;
    delete[] x_train;
    delete[] y_train;
    return 0; // Exit the program
}
