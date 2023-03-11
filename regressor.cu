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

__global__ void MVMult(float* matrix, float* vector, float* result, int M, int N, float bias ,float factor) 
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) 
	{	
		for (int i = 0; i < N; i++) 
		{
			result[row] += matrix[row * N + i] * vector[i] * factor;
		}
        result[row] += bias;
	}
}

__global__ void VVSub(float* vec1, float* vec2, float* res, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N)
    {
        res[index] = vec1[index] - vec2[index];
    }
}

__global__ void VCMult(float* vec, float num, float* res, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N)
    {
        res[index] = vec[index] * num;
    }
}

class Regressor
{ // Define Regressor class
private:
    float *theta, *d_x, *d_y, *d_theta; // Declare pointers for theta, x, y, and theta on device
    int m, n;                           // Declare number of training examples and features

    void transpose(float *x_train, float *x_train_transpose) 
    {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                x_train_transpose[j * n + i] = x_train[i * n + j];
            }
        }
    } 

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
        float *x_train_transpose = new float[n * m];
        transpose(x_train, x_train_transpose);

        float *d_x_train, *d_x_train_transpose,
                *d_y_train, *d_y_pred, *d_diff,
                *d_theta, *d_delta_theta;
        cudaMalloc(&d_x_train, m * n * sizeof(float));
        cudaMalloc(&d_y_train, m * sizeof(float));
        cudaMalloc(&d_theta, n * sizeof(float));

        // Copy the training data and initial parameters from host to device
        cudaMemcpy(d_x_train, x_train, m * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_train_transpose, x_train_transpose, n * m * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_train, y_train, m * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_theta, theta, n * sizeof(float), cudaMemcpyHostToDevice);
        
        int block_size = 1024;
        int grid_size = (m + block_size - 1) / block_size;
        
        float bias = 0;
        for (int i = 0; i < iterations; i++) 
        {
            // calculate y_pred
            cudaMalloc(&d_y_pred, m * sizeof(float));            
            MVMult<<<grid_size, block_size>>>(d_x_train, d_theta, d_y_pred, m, n, bias, 1);
            
            float *diff = new float[n];
            cudaMalloc(&d_diff, m * sizeof(float));
            cudaDeviceSynchronize();
            VVSub<<<grid_size, block_size>>>(d_y_pred, d_y_train, d_diff, m);
            cudaMemcpy(diff, d_diff, n * sizeof(float), cudaMemcpyDeviceToHost);
            
            cudaMalloc(&d_delta_theta, n * sizeof(float));
            cudaDeviceSynchronize();
            MVMult<<<1, n>>>(d_x_train_transpose, d_diff, d_delta_theta, n, m, 0, 1.0 / m);
            
            float sum = 0;
            for (int j = 0; j < n; j++) 
            {
                sum += diff[j];
            }
            float delta_bias = sum / m;
            bias -= alpha * delta_bias;
            
            cudaDeviceSynchronize();
            VCMult<<<1, n>>>(d_delta_theta, alpha, d_theta, n);
            cudaDeviceSynchronize();
            
            // clean up memory
            cudaFree(d_delta_theta);
            cudaFree(d_y_pred);
            cudaFree(d_diff);
            delete[] diff;
        }
        
        // Copy the final parameters from device back to host
        cudaMemcpy(theta, d_theta, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        // clean up memory
        cudaFree(d_x_train);
        cudaFree(d_x_train_transpose);
        cudaFree(d_y_train);
        cudaFree(d_theta);
        delete[] x_train_transpose;
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

    // example training data
    float *x_train = parseCSV("x_train.csv", m, n);
    float *y_train = parseCSV("y_train.csv", y_trainM, y_trainN);

    // example test data
    float *x_test = parseCSV("x_test.csv", x_testM, x_testN);

    // train model
    Regressor regressor(m, n);                              // Create a new instance of the Regressor class with m and n
    float alpha = 0.01;                                     // Set the learning rate alpha
    int iterations = 1000;                                  // Set the number of training iterations
    regressor.fit(x_train, y_train, alpha, iterations); // Fit the model to the training data

    // test model
    float y_pred = regressor.predict(x_test);                             // Predict the output for the test data point
    cout << "Predicted value: " << y_pred << endl;                        // Print the predicted y value

    delete[] x_test;
    delete[] x_train;
    delete[] y_train;
    return 0; // Exit the program
}