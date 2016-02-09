/* Program for finding out majority element in an array */
#include<stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include "timer.h"

#define bool int
#define EPS 0.000001
#define MAX_LENGTH_BLOCK 40
// MAX_DATA_BLOCK_ON_THREAD_BLOCK
#define MAX_DB_ON_TB 32

inline void cudaCheck(const cudaError_t &err, const std::string &mes) {
	if (err != cudaSuccess) {
		std::cout << (mes + " - " + cudaGetErrorString(err)) << std::endl;
		exit(EXIT_FAILURE);
	}
}

__device__ __host__ bool floatEquals(float a, float b) {
    if (fabs(a - b) < EPS) {
        return true;
    } else {
        return false;
    }
}

/* Function to find the candidate for Majority */
float findCandidate(const float * const a, int size) {
    int maj_index = 0, count = 1;
    int i;
    for(i = 1; i < size; i++) {
        if(floatEquals(a[maj_index], a[i])) {
            count++;
        } else {
            count--;
        }
        if(count == 0) {
            maj_index = i;
            count = 1;
        }
    }
    return a[maj_index];
}

/* Function to check if the candidate occurs more than n/2 times */
bool isMajority(const float * const a, int size, float cand) {
    int i, count = 0;
    for (i = 0; i < size; i++) {
        if(floatEquals(a[i], cand)) {
            count++;
        }
    }
    if (count > size/2) {
       return true;
    } else {
       return false;
    }
}

/* Function to print Majority Element */
int findMajority(const float * const a, int size) {
    /* Find the candidate for Majority*/
    float cand = findCandidate(a, size);

    if(isMajority(a, size, cand)) {
        return cand;
    } else {
        return -1;
    }
}

// This kernel is optimized for the maximum dimension
// of the input data is 100 count of blocks and 60 length of block.
__global__ void findMajorityKernel(
    const float * const data,
    int *results,
    const int length,
	const int countBlocks
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float dataOfBlockShared[MAX_LENGTH_BLOCK * MAX_DB_ON_TB];
	for (int i = 0; i < length; i += 1) {
		dataOfBlockShared[i * blockDim.x + threadIdx.x] = data[blockIdx.x * MAX_DB_ON_TB + i * countBlocks + threadIdx.x];
	}
	__syncthreads();
	const float * const dataOfBlock = &dataOfBlockShared[threadIdx.x];
	if (index < countBlocks) {
        int maj_index = 0;
        int count = 1;
        for(int i = 1; i < length; i++) {
            if(floatEquals(dataOfBlock[maj_index * MAX_DB_ON_TB], dataOfBlock[i * MAX_DB_ON_TB])) {
                count++;
            } else {
                count--;
            }
            if(count == 0) {
                maj_index = i;
                count = 1;
            }
        }
        int cand = dataOfBlock[maj_index * MAX_DB_ON_TB];

        int countCheck = 0;
        for (int i = 0; i < length; i++) {
            if(floatEquals(dataOfBlock[i * MAX_DB_ON_TB], cand)) {
                countCheck++;
            }
        }
		int result = -1;
        if (countCheck > length / 2) {
           result = cand;
	    }
		results[index] = result;
    }
}

/* Driver function to test above functions */
int main(int argc, char * argv[]) {
	try {
	    int length = 10;
	    int countBlocks = 1;
	    int countIter = 1000;
	    if  (argc > 1) {
	        length = atoi(argv[1]);
	    }
	    if (argc > 2) {
	        countBlocks = atoi(argv[2]);
	    }
	    const int sizeOfData = length * countBlocks;
		const int sizeOfDataBytes = length * countBlocks * sizeof(float);
	    std::vector<float> data(sizeOfData);
	    for (int i = 0; i < sizeOfData; ++i) {
	        // data[i] = rand() % (length / 3);
			data[i] = rand() % 2;
	        // std::cout << data[i] << " ";
	        // if (i % length == (length - 1)) {
	        //     std::cout << std::endl;
	        // }
	    }
	    // std::cout << std::endl;

	    // cuda implementation
	    Timer timer;
	    cudaError_t err = cudaSuccess;
	    float *dataDev;
	    err = cudaMalloc((void **)&dataDev, sizeOfDataBytes);
	    cudaCheck(err, "failed to allocated dataDev");
	    int *resultGPUDev;
	    err = cudaMalloc((void **)&resultGPUDev, countBlocks * sizeof(int));
	    cudaCheck(err, "failed to allocated resultGPUDev");

		std::vector<int> resultGPU(countBlocks);
	    float computeTimeGPU = 0.0f;
	    float computeTimeWithCopyGPU = 0.0f;
	    for (int i = 0; i < countIter; ++i) {
	        timer.begin("with copy");
			timer.begin("change data");
			std::vector<float> data1(sizeOfData);
			for (int j = 0; j < countBlocks; ++j) {
				for (int k = 0; k < length; ++k) {
					data1[k * countBlocks + j] = data[j * length + k];
				}
			}
			timer.end("change data");
			err = cudaMemcpy(dataDev, &data1[0], sizeOfDataBytes, cudaMemcpyHostToDevice);
			cudaCheck(err, "failed to copy data to the GPU");

	        int threadsPerBlock = 32;
			int blocksPerGrid = (countBlocks + threadsPerBlock - 1) / threadsPerBlock;
			timer.begin("compute");
			findMajorityKernel<<<blocksPerGrid, threadsPerBlock>>>(dataDev, resultGPUDev, length, countBlocks);
			cudaDeviceSynchronize();
			timer.end("compute");
			err = cudaGetLastError();
			cudaCheck(err, "failed to launch kernel");

			err = cudaMemcpy(&resultGPU[0], resultGPUDev, countBlocks * sizeof(int), cudaMemcpyDeviceToHost);
			cudaCheck(err, "failed to copy resultGPUDev to host");
			timer.end("with copy");
			computeTimeGPU += timer.getTimeMillisecondsFloat("compute") + timer.getTimeMillisecondsFloat("change data");
			computeTimeWithCopyGPU += timer.getTimeMillisecondsFloat("with copy");
	    }

		int countOfSucccesSearch = 0;
		float computeTimeCPU = 0.0f;
	    std::vector<int> resultCPU(countBlocks);
		for (int j = 0; j < countIter; ++j) {
		    for (int i = 0; i < countBlocks; ++i) {
				timer.begin("compute");
		        int result = findMajority(&data[length * i], length);
					resultCPU[i] = result;
				if (result != -1 && j == 0) {
					countOfSucccesSearch++;
				}
				timer.end("compute");
				computeTimeCPU += timer.getTimeMillisecondsFloat("compute");
		    }
		}
		std::cout << "count of success search " << countOfSucccesSearch << std::endl;
	    // for (auto &result : resultCPU) {
	    //     std::cout << result << " ";
	    // }
		// std::cout << std::endl;
		// for (auto &result : resultGPU) {
	    //     std::cout << result << " ";
	    // }
	    // std::cout << std::endl;

		// check correct work
		for (int i = 0; i < countBlocks; ++i) {
			if (resultCPU[i] != resultGPU[i]) {
				std::cout << "gpu and cpu results isn't equal." << std::endl;
				break;
			}
		}

		const int countOperations = countBlocks * log2f(length);
		const float avgComputeTimeGPU = computeTimeGPU / countIter;
		const float avgComputeTimeWithCopyGPU = computeTimeWithCopyGPU / countIter;
		std::cout << "avg compute time GPU = " << avgComputeTimeGPU << " milliseconds" << std::endl;
		std::cout << "avg compute time(with copy) GPU = " << avgComputeTimeWithCopyGPU << " milliseconds" << std::endl;
		std::cout << "Computational throughput GPU = " << countOperations / (avgComputeTimeGPU * 10e3) << " B/s" << std::endl;
		std::cout << "Computational throughput(with copy) GPU = " << countOperations / (avgComputeTimeWithCopyGPU * 10e3) << " B/s" << std::endl;

		const float avgComputeTimeCPU = computeTimeCPU / countIter;
		std::cout << "avg compute time CPU = " << avgComputeTimeCPU << " milliseconds" << std::endl;
		std::cout << "Computational throughput CPU = " << countOperations / (avgComputeTimeCPU * 10e3) << " B/s" << std::endl;
	} catch (const std::string &mes) {
		std::cout << "An error is occured - " << mes << std::endl;
	}

    // getchar();
    return 0;
}
