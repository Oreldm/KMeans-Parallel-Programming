
#include "Cuda_Parallel.h"
#include <cmath>

#define DIMENSIONS 2
#define GPU_DEVICE_ZERO 0

__global__ void distanceArrCalc(int pointsCounter, int threads, double *distanceFromPointToCluster, double *pointsInGpu, double *clustersInGpu)
{
	/**
	This Function computes distances. Every index is a point. Every value inside an index is a distance.
	**/
	double distanceX = 0;
	double distanceY= 0;

	int threadsLeft=pointsCounter % blockDim.x;
	if ((threadsLeft > threadIdx.x) || (blockIdx.x+1 != gridDim.x)) 
	{ 
		int offsetPointIndex=(blockIdx.x * threads + threadIdx.x)*DIMENSIONS;
		int offsetClusterIndexForPoint=threadIdx.y * DIMENSIONS;

		//calc X
		double a=pointsInGpu[offsetPointIndex];
		double b= clustersInGpu[offsetClusterIndexForPoint];
		distanceX = (a - b);
		distanceX*=distanceX;
		
		//calc Y
		a=pointsInGpu[offsetPointIndex+1];
		b= clustersInGpu[offsetClusterIndexForPoint+1];
		distanceY =(a - b);
		distanceY*=distanceY;


		double totalDistance=sqrt(distanceY+distanceX);
		int currentPointIndexY = pointsCounter*threadIdx.y;
		int currentPointIndexX=(blockIdx.x * threads + threadIdx.x);
		int pointIndex=currentPointIndexY+currentPointIndexX;

		distanceFromPointToCluster[pointIndex] = totalDistance;
	}
}

__global__ void minimumClusterDistance(int threads, double *pointToClusterDistance, int *minimumPointToCluster, int pointsCounter, int clusterCounter)
{
	/**
	This function puts the point in the right cluster after computing smallest distances.
	**/
	
	int leftThreads=pointsCounter % blockDim.x;

	if ((blockIdx.x +1 != gridDim.x) || (leftThreads > threadIdx.x)) 
	{ 
		int index=0;
		double smallestIndex; //minimum index
		double min; //minimum distance
		double temp; //temp distance
		int pointIndex=threads * blockIdx.x + threadIdx.x;
		min = pointToClusterDistance[pointIndex];
		int currentIndex;

		while(index<clusterCounter)
		{
			
			currentIndex=index*pointsCounter;
			temp = pointToClusterDistance[pointIndex + currentIndex];	
			if(temp < min)
			{
				smallestIndex = index;
				min = temp;
			}
			index++;
		}

		minimumPointToCluster[pointIndex] = smallestIndex;
	}
}

__global__ void pointToThreadMove(int pointsCounter, int threadsInsideBlock, double dt, double *pointsInGpu, double *speedArrayInGpu)					
{
	/**
	This function moves the thread with the right velocity readed from the file.
	This function puts every point in ONE thread.
	**/
	int blockDimLeft=pointsCounter % blockDim.x;
	if (blockIdx.x != gridDim.x - 1 || blockDimLeft > threadIdx.x)
	{
		int indexInArray=0;
		while(indexInArray < DIMENSIONS)
		{
			int currentBlock=blockIdx.x * DIMENSIONS * threadsInsideBlock;
			int currentThread=threadIdx.x* DIMENSIONS;
			int currentGpuPoint = currentBlock + currentThread + indexInArray;
			pointsInGpu[currentGpuPoint] += speedArrayInGpu[currentGpuPoint] * dt;		
			indexInArray++;
		}	
	}
}

void movePoints(int pointsCounter, double dt, double **pointsArr, double *pointsInGpu, double *speedArrayInGpu)
{
	/**
	This function moves all the points. It uses pointToThreadMove with <<Blocks,Threads>>  [every block have Threads threads].
	So it send every point to another thread.
	**/
	
	int blocks;
	int threads;

	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, GPU_DEVICE_ZERO);
	threads = properties.maxThreadsPerBlock;
	
	if(pointsCounter % threads == 0)
		blocks = pointsCounter / threads;
	else
		blocks = pointsCounter / threads + 1;

	pointToThreadMove<<<blocks, threads>>>(pointsCounter, threads, dt,pointsInGpu,speedArrayInGpu);
	cudaDeviceSynchronize();
	int sizeOfTotalPoint=pointsCounter * DIMENSIONS * sizeof(double);
	cudaMemcpy((void**)pointsArr[0], pointsInGpu, sizeOfTotalPoint, cudaMemcpyDeviceToHost);
}

void pointsToCluster(int pointsCounter, int clusterCount, int *pointToCluster, double *pointsInGpu, double **clusterArr)
{
	/**
	This function change every point to the right cluster and sends it back to the CPU.
	It uses 2 simple steps - parallel functions:
	1. distanceArrCalc - to calculate distance to every point
	2. minimumClusterDistance - calssify the point to the right cluster
	**/
	int blocks, threads, *finalPointToClusterMapping;
	double *clusters , *pointToClusterArr;

	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, GPU_DEVICE_ZERO);
	threads = properties.maxThreadsPerBlock / clusterCount;
	dim3 threadTable(threads, clusterCount);

	initializeThreadsBlocks(&threads, &blocks, pointsCounter);

	cudaMalloc((void**)&finalPointToClusterMapping, pointsCounter*sizeof(int));
	int arraylenth=DIMENSIONS * clusterCount * sizeof(double);
	cudaMalloc((void**)&clusters, arraylenth);
	cudaMalloc((void**)&pointToClusterArr, arraylenth);

	cudaMemcpy(clusters, clusterArr[0], clusterCount * DIMENSIONS * sizeof(double), cudaMemcpyHostToDevice);
	distanceArrCalc <<<blocks, threadTable>>> (pointsCounter, threads, pointToClusterArr,pointsInGpu,clusters);
	cudaDeviceSynchronize();

	threads = properties.maxThreadsPerBlock;
	initializeThreadsBlocks(&threads, &blocks, pointsCounter);
	minimumClusterDistance <<<blocks, threads>>> (threads, pointToClusterArr, finalPointToClusterMapping,pointsCounter , clusterCount);
	cudaDeviceSynchronize();
	cudaMemcpy(pointToCluster, finalPointToClusterMapping, pointsCounter * sizeof(int), cudaMemcpyDeviceToHost);

}

void initializeThreadsBlocks(int *threads, int* blocks, int pointsCounter)
{
	/**
	This function initialize the block size.
	Sometimes the are too many threads (because of points%threads > 0 ) so we need to do it.
	If it is bigger, we will add 1 to the block size.
	Every thread will add 1 so it will be exactly the right size.
	**/
	if (pointsCounter % *threads == 0) 
	{ 
		*blocks = pointsCounter / *threads; 
	}
	else
	{
		*blocks= pointsCounter / *threads + 1;
	}

}
