
#include "../libwb/wb.h"
#define BLOCK_WIDTH 16
#define wbCheck(stmt) \
    do { \
        cudaError_t err = stmt; \
        if (err != cudaSuccess) { \
            wbLog(ERROR, "Failed to run stmt ", #stmt); \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
            return -1; \
        } \
    } while(0) \

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                                int numAColumns, int numBRows,
                                int numBColumns, int numCRows,
                                int numCColumns) {
                                    // @@ Insert code to implement matrix multiplication here
                                    int nCol = blockIdx.y * blockDim.y + threadIdx.y;
                                    int nRow = blockIdx.x * blockDim.x + threadIdx.x;

                                    if (nRow < numCRows && nCol < numCColumns) { 
                                        float fCval = 0.0f;
                                        for (int i = 0; i < numBRows; ++i) {
                                            fCval += A[numAColumns * nRow + i] * B[i * numBColumns + nCol];
                                        }
                                        C[nRow * numCColumns + nCol] = fCval;
                                    }
                                }

int main(int argc, char **argv) {
    wbArg_t args;
    float *hostA; // The A matrix
    float *hostB;
    float *hostC;
    float *deviceA;
    float *deviceB;
    float *deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix B
    int numBRows;
    int numBColumns;
    int numCRows;
    int numCColumns;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float*)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
    hostB = (float*)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
    
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;

    // @@ Allocate the hostC matrix
    hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
    wbTime_stop(Generic, "Import data and creating memory on host");

    wbLog(TRACE, "The dimension of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimension of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    // @@ Allocate GPU memory here
    cudaMalloc((void **) &deviceA, numARows * numAColumns * sizeof(float));
    cudaMalloc((void **) &deviceB, numBRows * numBColumns * sizeof(float));
    cudaMalloc((void **) &deviceC, numARows * numBColumns * sizeof(float));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here

    cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    // @@ Initialize the grid and block dimension here
    
    dim3 Grid(ceil((float)numCRows / BLOCK_WIDTH), ceil((float)numCColumns / BLOCK_WIDTH), 1);
    dim3 Block(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    wbTime_start(Compute, "Performing CUDA computation");
    // @@ Launch the GPU Kernel her
    matrixMultiply<<<Grid, Block>>>(deviceA, deviceB, deviceC, 
                                    numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Perfoming CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    // @@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    // @@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}