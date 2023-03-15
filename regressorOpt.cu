/**
 * Anthony Bustamante, Jesse Leu, Khushaal Kurswani
 * CSS 535 High Performance Computing
 * Professor Erika Parsons
 * 16 March 2023
 *
 * Final project - Accelerating Regression Model Training using GPU
 * Optipmized code
 *
 * Compile in CLI using the following command:
 *      nvcc regressor.cu
 *
 * To profile the kernel functions, nsight compute or nvprof can be used
 * nsight compute CLI command:
 *      ncu -o <profiler_output_file_name>  --set full <executable_file>
 *
 */
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
#include <iomanip>  // setw
using namespace std;
#define MAX_SHARE_SIZE 12000
const int unroll_factor = 4;

/**
 *@brief This is a CUDA kernel function that performs matrix-vector multiplication
 *@param matrix  input matrix
 *@param vector  input vector array
 *@param result  output vector array
 *@param M the number of rows
 *@param N the number of columns
 *@param bias a scalar value added to each element in the output vector
 *@param factor a scalar value multiplied to each element in the output vector
 *@param numShared number of element should be in the share memory
 */

__global__ void MVMult(float *matrix, float *vector, float *result, int M, int N, float bias, float factor, int numShared)
{
__shared__ float cachedVector[MAX_SHARE_SIZE];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int numRowsPerThread = numShared / blockDim.x;
    int startIndex = threadIdx.x * numRowsPerThread;
    int endIndex = startIndex + numRowsPerThread - 1;
    for (int i = startIndex; i <= endIndex; i++)
    {
        cachedVector[i] = vector[i];
    }
    int numCopied = blockDim.x * numRowsPerThread;

    __syncthreads();
    if (row < M)
    {
        int i = 0;
        for (; i < numCopied && i + 3 < N; i+=4)
        {
            result[row] += matrix[row * N + i] * cachedVector[i];
            result[row] += matrix[row * N + i + 1] * cachedVector[i + 1];
            result[row] += matrix[row * N + i + 2] * cachedVector[i + 2];
            result[row] += matrix[row * N + i + 3] * cachedVector[i + 3];
        }

        for (; i < N; i++)
        {
            result[row] += matrix[row * N + i] * vector[i];
        }

        result[row] *= factor;
        result[row] += bias;
    }
}

/**
 *@brief This CUDA kernel function performs vector-vector subtraction
 *@param vec1  the first input vector array
 *@param vec2  the second input vector array
 *@param res  the output vector array
 *@param N the length of the input vectors
 */
__global__ void VVSub(float *vec1, float *vec2, float *res, int N)
{
    int index = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (index < N && index + 3 < N) // loop unrolling
    {
        res[index] = vec1[index] - vec2[index];
        res[index + 1] = vec1[index + 1] - vec2[index + 1];
        res[index + 2] = vec1[index + 2] - vec2[index + 2];
        res[index + 3] = vec1[index + 3] - vec2[index + 3];
    }
}

/**
 *@brief This CUDA kernel function performs vector-vector subtraction on the left-over
 *elements of input vectors after VVSub kernel has processed multiple of 4 elements.
 *@param vec1  the first input array
 *@param vec2 the second input array
 *@param res  the output array
 *@param N the length of the input vectors
 *@param offset the starting index in the input vectors for this kernel's processing
 */
__global__ void VVSubLeftOver(float *vec1, float *vec2, float *res, int N, int offset)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x + offset;
    if (index < N)
    {
        res[index] = vec1[index] - vec2[index];
    }
}

/**
 *@brief This is a CUDA kernel function that performs vector-constant multiplication
 *@param vec  input vector array
 *@param num the constant scalar value to multiply
 *@param res  the output array
 *@param N the length of the input and output vectors
 */
__global__ void VCMult(float *vec, float num, float *res, int N)
{
    int index = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (index < N && index + 3 < N) // loop unrolling
    {
        res[index] -= vec[index] * num;
        res[index + 1] -= vec[index + 1] * num;
        res[index + 2] -= vec[index + 2] * num;
        res[index + 3] -= vec[index + 3] * num;
    }
}

/**
 *@brief This is a CUDA kernel function that performs vector-constant multiplication
 *for the leftover elements in case the length of the vector is not a multiple of 4.
 *@param vec the input vector array
 *@param num the constant scalar value to multiply
 *@param res the output array
 *@param N the length of the input and output vectors
 *@param offset the number of elements already processed by previous kernels
 */
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
    int m, n;               // Declare number of training examples and features

    /**
     *@brief This function transposes a given 2D matrix of dimensions m x n
     *and stores the result in a new 2D matrix of dimensions n x m.
     *@param x_train a pointer to the input matrix array
     *@param x_train_transpose a pointer to the output matrix array
     */
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

    /**
     *@brief This function trains a linear regression model using batch gradient descent optimization
     *on the given training data (x_train, y_train). It uses CUDA parallelism to accelerate
     *the matrix and vector computations. The optimization algorithm runs for the specified
     *number of iterations, and updates the model parameters (theta) in each iteration. The
     *learning rate (alpha) and block sizes (mv_block_size and vv_block_size) can also be
     *specified. The function returns void, but updates the theta parameter in-place.
     *@param x_train  array of training data input features
     *@param y_train array of training data output labels
     *@param alpha Learning rate for the gradient descent optimization
     *@param iterations Number of iterations to run the optimization algorithm
     *@param mv_block_size Block size to use for matrix-vector multiplication kernel
     *@param vv_block_size Block size to use for vector-vector subtraction kernel
     */
    void fit(float *x_train, float *y_train, float alpha, int iterations,
             int mv_block_size, int vv_block_size)
    {
        float *x_train_transpose = new float[n * m];
        transpose(x_train, x_train_transpose);

        float *d_x_train, *d_x_train_transpose,
            *d_y_train, *d_y_pred, *d_diff,
            *d_theta, *d_delta_theta;

        // Allocate kernel memory
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
            int numShare = m;
            if(numShare > MAX_SHARE_SIZE){
                numShare = MAX_SHARE_SIZE;
            }
            MVMult<<<mv_grid_size, mv_block_size>>>(d_x_train, d_theta, d_y_pred, m, n, bias, 1, numShare);

            float *diff = new float[m];
            cudaMalloc(&d_diff, m * sizeof(float));
            // calculate configuration for vector vector sub operation
            int vv_elementsPerBlock = vv_block_size * unroll_factor;
            int vv_grid_size = m / vv_elementsPerBlock;

            // calculate configuration for vector vector sub operation for left over
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
            MVMult<<<1, n>>>(d_x_train_transpose, d_diff, d_delta_theta, n, m, 0, 1.0 / m, numShare);

            float sum = 0;
            for (int j = 0; j < n; j++)
            {
                sum += diff[j];
            }
            float delta_bias = sum / m;
            bias -= alpha * delta_bias;

            // calculate configuration for vector constant operation
            int vc_block_size = min(n / unroll_factor, 1024);
            int vc_elementsPerBlock = vc_block_size * unroll_factor;
            int vc_grid_size = n / vc_elementsPerBlock;

            cudaDeviceSynchronize();
            VCMult<<<vc_grid_size, vc_block_size>>>(d_delta_theta, alpha, d_theta, n);

            // handle the vector constant operation left over
            int vc_leftOver = n - vc_grid_size * vc_elementsPerBlock;
            if (vc_leftOver > 0)
            {
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

    /**
     *@brief Predicts the output values for a given set of input features using the learned model parameters.
     *@param x_test the input features to be predicted
     *@param size Number of samples in the input features.
     *@return Pointer to an array containing the predicted output values.
     */
    float *predict(float *x_test, int size)
    {
        float *y_pred = new float[size];

        // Calculate the predicted value using the learned parameters
        for (int i = 0; i < size; i++)
        {
            y_pred[i] = 0;
            for (int j = 0; j < n; j++)
            {
                y_pred[i] += theta[j] * x_test[i * n + j];
            }
            y_pred[i] *= 10;
        }

        // Return the predicted values
        return y_pred;
    }

    /**
     *@brief  This function prints the learned weights of the linear regression model to the console
     */
    void printWeights()
    {
        for (int i = 0; i < n; i++)
        {
            printf("Feature[%d] Weight: %.6f ", i, theta[i]);
            
        }
        cout << endl;
    }
};
/**
 *@brief  Parses a CSV file containing float values and returns them in a 1D array.
 *@param fName A string representing the name of the CSV file to be parsed.
 *@param m the number of rows in the CSV file.
 *@param n  the number of columns in the CSV file.
 *@return flatten matrix of floats containing the values from the CSV file.
 */
float *parseCSV(string fName, int &m, int &n)
{
    ifstream data(fName);
    string line;
    string item;
    vector<vector<float>> dataMatrix;
    getline(data, line);        // skip the title
    while (getline(data, line)) // readline
    {
        vector<float> dataVec;
        string data;
        stringstream lineStream(line);

        while (getline(lineStream, item, ',')) // split
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
/**
 *@brief Prints a 2D matrix of floats (frist 10 X 10 )
 *@param A  a flatten matrix
 *@param m  number of rows in the matrix.
 *@param n  number of columns in the matrix.
 */
void printMatrix(float *A, int m, int n)
{
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << A[i * n + j] << " ";
            if(j == 10){
                cout << "..." <<endl;
                break;
            }
            
        }
        cout <<endl;
    }
    cout << "..."<< endl;
}

/**
 *@brief  Find min, max, avg, and range in a col of a matrix.
 *@param A  a flatten matrix
 *@param m  number of rows in the matrix.
 *@param n  number of columns in the matrix.
 *@param col the target col index
 *@param min the min nuumber in the col
 *@param max the max nuumber in the col
 *@param avg the avg nuumber in the col
 *@param range the max - min in the col
 */
void findStat(float *A, int m, int n, int col, float &min, float &max, float &avg, float &range)
{
    float sum = 0;

    for (int i = 0; i < m; i++)
    {
        sum += A[i * n + col];

        if (A[i * n + col] < min)
        { // check min
            min = A[i * n + col];
        }
        if (A[i * n + col] > max)
        { // check max
            max = A[i * n + col];
        }
    }
    avg = sum / m;
    range = max - min;
}
/**
 *@brief  normalize a col in a matrix
 *@param A  a flatten matrix
 *@param m  number of rows in the matrix.
 *@param n  number of columns in the matrix.
 *@param col the target col index
 */
void normalizeCol(float *A, int col, int m, int n)
{
    float range = -1, max = std::numeric_limits<float>::min(),
          min = std::numeric_limits<float>::max(), avg = 0;

    findStat(A, m, n, col, min, max, avg, range);
    for (int i = 0; i < m; i++)
    {
        A[i * n + col] = (A[i * n + col] - min) / range; // update to normalize val
    }
}
/**
 *@brief  normalize a matrix feature by feature
 *@param A  a flatten matrix
 *@param m  number of rows in the matrix.
 *@param n  number of columns in the matrix.
 */
void normalizeAllByFeature(float *A, int m, int n)
{
    for (int i = 0; i < n; i++)
    {
        normalizeCol(A, i, m, n);
    }
}

/**
 *@brief Calculates the flops of operations.
 *@param elapsed A float representing the elapsed time operation.
 *@param m the number of rows in the matrix.
 *@param n the number of columns in the matrix.
 *@param iterations An integer representing the number of iterations performed.
 *@return A float representing the FLOPS
 */
float calcFLOPS(float elapsed, int m, int n, int iterations)
{
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

    FLOP *= iterations;

    return FLOP / elapsed;
}

/**
 *@brief Performs test with varying block sizes for training the model.
 *@param m the number of rows in input matrix.
 *@param n the number of columns in input matrix.
 *@param alpha learning rate
 *@param x_train   x_train flatten matrix
 *@param y_train  y_train flatten matrix
 */
void blocksExperiment(int m, int n, float alpha, float *x_train, float *y_train)
{
    int numSizes = 4;
    int blockSizeList[numSizes] = {128, 256, 512, 1024};

    Regressor regressor(m, n);

    for (int i = 0; i < numSizes; i++)
    {
        regressor.fit(x_train, y_train, alpha, 1, blockSizeList[i], blockSizeList[i]);
    }
}
/**
 *@brief Trains a regression model using the input matrix and vector data, and returns the trained model.
 *@param m the number of rows in the input matrix.
 *@param n the number of columns in the input matrix.
 *@param alpha learning rate
 *@param iterations  the number of iterations to be performed during training.
 *@param x_train x_train
 *@param y_train y_train
 *@return A Regressor object representing the trained regression model.
 */
Regressor trainRegressor(int m, int n, float alpha, float iterations, float *x_train, float *y_train)
{
    Regressor regressor(m, n);

    clock_t startTraining = clock();                                // start training timer
    regressor.fit(x_train, y_train, alpha, iterations, 1024, 1024); // Fit the model to the training data
    clock_t endTraining = clock();                                  // start training timer

    float elapsedTraining = (endTraining - startTraining) / (CLOCKS_PER_SEC / pow(10, 3));
    cout << "Training Time: " << elapsedTraining << " milliseconds" << endl;

    float trainingFLOPS = calcFLOPS(elapsedTraining / 1000, m, n, iterations);
    cout << "Training FLOPS: " << trainingFLOPS << " FLOPS" << endl;

    return regressor;
}
/**
 *@brief Calculates the R-squared value for a given set of true and predicted values.
 *@param y_test an array of true target values
 *@param y_pred an array of predicted target values
 *@param n the length of the input arrays
 *@return the R-squared value
 */
float r_squared(float y_test[], float y_pred[], int n)
{
    float y_mean = accumulate(y_test, y_test + n, 0.0f) / n;
    float ss_tot = 0.0f, ss_res = 0.0f;
    for (int i = 0; i < n; i++)
    {
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
    cout << "Top 10 rows of x_train: " << endl;
    printMatrix(x_train, m, n);
    normalizeAllByFeature(x_train, m, n);
    float *y_train = parseCSV("y_train.csv", y_trainM, y_trainN);

    // example test data
    float *x_test = parseCSV("x_test.csv", x_testM, x_testN);
    normalizeAllByFeature(x_test, x_testM, x_testN);

    float *y_test = parseCSV("y_test.csv", y_testM, y_testN);

    float alpha = 0.01;    // Set the learning rate alpha
    int iterations = 1000; // Set the number of training iterations

    // block experiment
    blocksExperiment(m, n, alpha, x_train, y_train);

    // train model
    Regressor regressor = trainRegressor(m, n, alpha, iterations, x_train, y_train);
    cout << "Regressor Model Weights: ";
    regressor.printWeights();

    // test model
    float *y_pred = regressor.predict(x_test, x_testM); // Predict the output for the test data point

    // Print predictions
    cout << "Predict Value:             Actual Value: " << endl;
    for (int i = 0; i < 10; i++){
        printf("predict[%d]:  %10.2f    Actual[%d]  %.2f \n", i, y_pred[i], i ,y_test[i]);
    }
    cout << "..."<< endl;

    cout << "Model r-squared value: " << r_squared(y_test, y_pred, y_testM) << endl;

    delete[] x_test;
    delete[] x_train;
    delete[] y_train;
    delete[] y_pred;
    delete[] y_test;
    return 0; // Exit the program
}