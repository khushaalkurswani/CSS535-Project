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
#include <limits>
#include <time.h>
#include <numeric>

using namespace std;

const int unroll_factor = 4;

__global__ void MVMult(float *matrix, float *vector, float *result, int M, int N, float bias, float factor)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M)
    {
        for (int i = 0; i < N; i++)
        {
            result[row] += matrix[row * N + i] * vector[i];
        }
        result[row] *= factor;
        result[row] += bias;
    }
}

__global__ void VVSub(float *vec1, float *vec2, float *res, int N)
{
    int index = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (index < N && index + 3 < N)
    {
        res[index] = vec1[index] - vec2[index];
        res[index + 1] = vec1[index + 1] - vec2[index + 1];
        res[index + 2] = vec1[index + 2] - vec2[index + 2];
        res[index + 3] = vec1[index + 3] - vec2[index + 3];
    }
}

__global__ void VVSubLeftOver(float *vec1, float *vec2, float *res, int N, int offset)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x + offset;
    if (index < N)
    {
        res[index] = vec1[index] - vec2[index];
    }
}

__global__ void VCMult(float *vec, float num, float *res, int N)
{
    int index = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (index < N && index + 3 < N)
    {
        res[index] -= vec[index] * num;
        res[index + 1] -= vec[index + 1] * num;
        res[index + 2] -= vec[index + 2] * num;
        res[index + 3] -= vec[index + 3] * num;
    }
}

__global__ void VCMultLeftOver(float *vec, float num, float *res, int N, int offset)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x + offset;
    if (index < N)
    {
        res[index] -= vec[index] * num;
    }
}

class Regressor
{ // Define Regressor class
private:
    float *theta, *d_theta; // Declare pointers for theta on host and device
    int m, n;                           // Declare number of training examples and features

    void transpose(float *x_train, float *x_train_transpose)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                x_train_transpose[j * m + i] = x_train[i * n + j];
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
        cudaFree(d_theta); // Free memory for theta on device
    }

    void fit(float *x_train, float *y_train, float alpha, int iterations,
        int mv_block_size, int vv_block_size)
    {
        float *x_train_transpose = new float[n * m];
        transpose(x_train, x_train_transpose);

        float *d_x_train, *d_x_train_transpose,
            *d_y_train, *d_y_pred, *d_diff,
            *d_theta, *d_delta_theta;
        cudaMalloc(&d_x_train, m * n * sizeof(float));
        cudaMalloc(&d_x_train_transpose, m * n * sizeof(float));
        cudaMalloc(&d_y_train, m * sizeof(float));
        cudaMalloc(&d_theta, n * sizeof(float));

        // Copy the training data and initial parameters from host to device
        cudaMemcpy(d_x_train, x_train, m * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_train_transpose, x_train_transpose, n * m * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_train, y_train, m * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_theta, theta, n * sizeof(float), cudaMemcpyHostToDevice);

        float bias = 0;
        for (int i = 0; i < iterations; i++)
        {
            // calculate y_pred
            int mv_grid_size = (m + mv_block_size - 1) / mv_block_size;
            cudaMalloc(&d_y_pred, m * sizeof(float));
            MVMult<<<mv_grid_size, mv_block_size>>>(d_x_train, d_theta, d_y_pred, m, n, bias, 1);
            
            float *diff = new float[m];
            cudaMalloc(&d_diff, m * sizeof(float));
            
            int vv_elementsPerBlock = vv_block_size * unroll_factor;
            int vv_grid_size = m / vv_elementsPerBlock;

            int vv_offset = vv_elementsPerBlock * vv_grid_size;

            int vv_leftOver = m - vv_grid_size * vv_elementsPerBlock;
            int vv_leftOver_block_size = 1024;
            int vv_leftOver_grid_size = (vv_leftOver + vv_leftOver_block_size - 1) / vv_leftOver_block_size;

            cudaDeviceSynchronize();
            VVSub<<<vv_grid_size, vv_block_size>>>(d_y_pred, d_y_train, d_diff, m);
            VVSubLeftOver<<<vv_leftOver_grid_size, vv_leftOver_block_size>>>(d_y_pred, d_y_train, d_diff, m, vv_offset);
            cudaMemcpy(diff, d_diff, m * sizeof(float), cudaMemcpyDeviceToHost);
            
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
            
            int vc_block_size = min(n / unroll_factor, 1024);
            int vc_elementsPerBlock = vc_block_size * unroll_factor;
            int vc_grid_size = n / vc_elementsPerBlock;
            
            cudaDeviceSynchronize();
            VCMult<<<vc_grid_size, vc_block_size>>>(d_delta_theta, alpha, d_theta, n);
            
            int vc_leftOver = n - vc_grid_size * vc_elementsPerBlock;
            if (vc_leftOver > 0) {
                int vc_offset = vc_elementsPerBlock * vc_grid_size;
                int vc_leftOver_block_size = min(vc_leftOver, 1024);
                int vc_leftOver_grid_size = (vc_leftOver + vc_leftOver_block_size - 1) / vc_leftOver_block_size;
                VCMultLeftOver<<<vc_leftOver_grid_size, vc_leftOver_block_size>>>(d_delta_theta, alpha, d_theta, n, vc_offset);
            }
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

    float *predict(float *x_test, int size)
    {
        float *y_pred = new float[size];

        // Calculate the predicted value using the learned parameters
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < n; j++)
            {
                y_pred[i] += theta[j] * x_test[i * n + j];
            }
        }

        // Return the predicted values
        return y_pred;
    }

    void printWeights() 
    {
        for (int i = 0; i < n; i++)
        {
            cout << theta[i] << ", ";
        }
        cout << endl;
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
void findStat(float *A, int m, int n, float &min, float &max, float &avg, float &range)
{
    float sum = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if(j == 0){
                continue;
            }
            sum += A[i * n + j];
            if (A[i * n + j] < min)
            {
                min = A[i * n + j];
            }
            if (A[i * n + j] > max)
            {
                max = A[i * n + j];
            }
        }
    }
    avg = sum / (m * n);
    range = max - min;
}

void normalizeAll(float *A, int m, int n)
{
    float range = -1, max = std::numeric_limits<float>::min(),
          min = std::numeric_limits<float>::max(), avg = 0;
          findStat(A, m, n, min, max, avg, range);
          for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(j == 0){
                    continue;
                }
                A[i * n + j] = (A[i * n + j] - min) / range;
            }
          }
}

 void findStat(float* A, int m, int n, int col,float& min, float& max, float& avg, float& range){
    float sum = 0;

    for(int i = 0; i < m; i++){
        sum += A[i * n + col];

        if(A[i * n + col] < min){
            min = A[i * n + col];
        }
        if(A[i * n + col] > max){
            max = A[i * n + col];

        }
    }
    avg = sum / m;
    range = max - min;

}

void normalizeCol(float* A, int col,int m, int n){
    float range = -1, max = std::numeric_limits<float>::min(),
    min = std::numeric_limits<float>::max(),avg = 0;

    findStat(A, m, n, col,min, max,avg,range);
    for(int i = 0; i < m; i++){
        A[i * n + col] = (A[i * n + col] - min) / range;
    }
}

void normalizeAllByFeature(float* A, int m, int n){
    for(int i = 0; i < n; i++){
        normalizeCol(A, i, m, n);
    }
}

float calcFLOPS(float elapsed, int m, int n) {
    // matrix vector dot product is 2mn and adding bias is m operations 
    //      (multiply by factor is m operations) --> Unecessary operation for 
    //      this step but part of MVMult kernel
    int FLOP = 2 * m * n + 2 * m; 

    // vector subtraction is m, matrix vector dot product is 2nm, and 
    //      multiply by factor is n operations (adding bias is n 
    //      operations) Unecessary operation for  this step but part of 
    //      MVMult kernel
    FLOP += m + 2 * n * m + n;

    // vector sum is m operations and vector constant multiplication m operations
    //      Reusing vector subtraction result from previous step so not included 
    //      in FLOP calculation for this step
    FLOP += 2 * m;

    // constant multiplication is 1 operation and constant substraction is 1 operation
    FLOP += 2;

    return FLOP / elapsed;
}

void blocksExperiment(int m, int n, float alpha, float *x_train, float *y_train) 
{
    int numSizes = 4;
    int blockSizeList[numSizes] = {128, 256, 512, 1024};

    Regressor regressor(m, n); 

    for (int i = 0; i < numSizes; i++) {
        regressor.fit(x_train, y_train, alpha, 1, blockSizeList[i], blockSizeList[i]);
    }

}

Regressor trainRegressor(int m, int n, float alpha, float iterations, float *x_train, float *y_train)
{
    Regressor regressor(m, n);
    
    clock_t startTraining = clock(); // start training timer
    regressor.fit(x_train, y_train, alpha, iterations, 1024, 1024); // Fit the model to the training data
    clock_t endTraining = clock(); // start training timer

    float elapsedTraining = (endTraining - startTraining) / (CLOCKS_PER_SEC / pow(10, 3));
	cout << "Training Time: " << elapsedTraining << " milliseconds" << endl;

    float trainingFLOPS = calcFLOPS(elapsedTraining / 1000, m, n);
    cout << "Training FLOPS: " << trainingFLOPS << " FLOPS" << endl;

    return regressor;
}

float r_squared(float y_test[], float y_pred[], int n) {
    float y_mean = accumulate(y_test, y_test + n, 0.0f) / n;
    float ss_tot = 0.0f, ss_res = 0.0f;
    for (int i = 0; i < n; i++) {
        ss_tot += pow(y_test[i] - y_mean, 2);
        ss_res += pow(y_test[i] - y_pred[i], 2);
    }
    float r2 = 1 - (ss_res / ss_tot);
    return r2;
}

int main()
{
    int m, n, y_trainM, y_trainN, x_testM, x_testN, y_testM, y_testN;

    // example training data
    float *x_train = parseCSV("x_train.csv", m, n);
    normalizeAll(x_train, m, n);
    normalizeCol(x_train, 0, m, n);
    float *y_train = parseCSV("y_train.csv", y_trainM, y_trainN);

    // example test data
    float *x_test = parseCSV("x_test.csv", x_testM, x_testN);
    normalizeAll(x_test, x_testM, x_testN);

    float *y_test = parseCSV("y_test.csv", y_testM, y_testN);

    float alpha = 0.01;                                 // Set the learning rate alpha
    int iterations = 1000;                              // Set the number of training iterations

    // block experiment
    blocksExperiment(m, n, alpha, x_train, y_train);                               

    // train model
        // Create a new instance of the Regressor class with m and n
    Regressor regressor = trainRegressor(m, n, alpha, iterations, x_train, y_train);
    cout << "Regressor Model Weights: ";
    regressor.printWeights();
     
    // test model
    int size = x_testM / n;
    float *y_pred = regressor.predict(x_test, size); // Predict the output for the test data point

    // Print predictions
    for (int i = 0; i < size; i++)
    {
        cout << y_pred[i] << endl;
    }

    cout << "Model r-squared value: " << r_squared(y_test, y_pred, size) << endl;

    delete[] x_test;
    delete[] x_train;
    delete[] y_train;
    delete[] y_pred;
    delete[] y_test;
    return 0; // Exit the program
}