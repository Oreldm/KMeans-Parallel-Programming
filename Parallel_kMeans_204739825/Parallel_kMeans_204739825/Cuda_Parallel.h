#ifndef __Cuda_Parallel_H_
#define __Cuda_Parallel_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


void movePoints( int pointsCounter, double dt, double** pointsArr, double* pointsInGpu, double* speedArrayInGpu);
	
void pointsToCluster(int pointsCounter, int clusterCount, int *pointToCluster, double *pointsInGpu, double **clusterArr);

void initializeThreadsBlocks(int *threads, int* blocks, int pointsCounter);

#endif