#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <omp.h>
#include "Cuda_Parallel.h"


#define ROOTID 0
#define DIMENSIONS 2
#define INT_ARRAY_SIZE_BROADCAST 4
#define DOUBLE_ARRAY_SIZE_BROADCAST 2
#define INPUT "C:\\input.txt"
#define OUTPUT "C:\\output.txt"
#define GPU_ZERO 0


void initializeMpi(int argc, char **argv, int* id, int* proccessesNumber);
int* allocateIntMemory(int size);
double* readPoints(FILE* file, int *N, int *K, int *T, double *dT, int *LIMIT, double *QM, double **velocity);
void writeClusters(FILE* file, double **clustersArray, int K, double T, double QM);
double* calculateDiameters(double *pointsArray, int N, int K, int *pointToClusterMapping);
double qualityCalculation(double **clustersArray, int clusterCount, double *clustersDiametersArray);
void kMeansAlgorithm(MPI_Comm comm, double *gpuArrayPoints, int N, int K, int LIMIT, int *pointsToClusterMapping, double **clustersArray, double **pointsArray);
int main(int argc, char *argv[])
{
	int proccessesNumber, currentId;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &currentId);
	MPI_Comm_size(MPI_COMM_WORLD, &proccessesNumber);

	if (proccessesNumber < 3) {
		printf("PLEASE START THE PROGRAM WITH AT LEAST THREE PROCCESSES");
		fflush(stdout);
		MPI_Finalize();
		return 0;
	}

	//Input file variables
	double T, dT, QM;
	int N, K, LIMIT;

	int index, numberOfPointsInProccess, maxTime;

	int *pointClusterMapping,
		*pointToCluster,
		*pointsToSend = NULL,
		*pointGatherArr = NULL,		
		*recvCountsUpdatedPoints,
		*gapsArr = NULL,
		*gapsGatherArr = NULL,	
		*updatedPointsGaps, 
		*intInitializeArrayBroadcast;

	double	 *pointsFromFile,
		*pointsArrayInGpu = NULL,
		*velocityArrayInGpu = NULL,
		**clustersArray,	
		qualityAtCurrentTime,		
		**proccessPoints,	
		*pointsVeloFromFile,		
		**proccessVelocity,		
		*doubleInitializeArrayBroadcast,		
		timeCounter = 0;		

	pointsToSend = allocateIntMemory(proccessesNumber);
	gapsArr = allocateIntMemory(proccessesNumber);
	pointGatherArr = allocateIntMemory(proccessesNumber);
	gapsGatherArr = allocateIntMemory(proccessesNumber);
	recvCountsUpdatedPoints = allocateIntMemory(proccessesNumber);
	updatedPointsGaps = allocateIntMemory(proccessesNumber);
	intInitializeArrayBroadcast = allocateIntMemory(INT_ARRAY_SIZE_BROADCAST);
	doubleInitializeArrayBroadcast = (double*)malloc(DOUBLE_ARRAY_SIZE_BROADCAST * sizeof(double));


	T = MPI_Wtime();

	if (currentId == ROOTID)
	{
		FILE* inputFile = fopen(INPUT, "r");
		
		pointsFromFile = readPoints(inputFile,&N,&K,&maxTime,&dT,&LIMIT,&QM,&pointsVeloFromFile);
		pointClusterMapping = (int*)malloc(N * sizeof(int));

		intInitializeArrayBroadcast[0] = N;
		intInitializeArrayBroadcast[1] = LIMIT;
		intInitializeArrayBroadcast[2] = K;
		intInitializeArrayBroadcast[3] = maxTime;

		doubleInitializeArrayBroadcast[0] = QM;
		doubleInitializeArrayBroadcast[1] = dT;
	}

	MPI_Bcast(intInitializeArrayBroadcast, INT_ARRAY_SIZE_BROADCAST, MPI_INT, ROOTID, MPI_COMM_WORLD);
	MPI_Bcast(doubleInitializeArrayBroadcast, DOUBLE_ARRAY_SIZE_BROADCAST, MPI_DOUBLE, ROOTID, MPI_COMM_WORLD);
	
	N = intInitializeArrayBroadcast[0];
	LIMIT = intInitializeArrayBroadcast[1];
	K = intInitializeArrayBroadcast[2];
	maxTime = intInitializeArrayBroadcast[3];

	QM = doubleInitializeArrayBroadcast[0];
	dT = doubleInitializeArrayBroadcast[1];

	int currentProccess =0, gap=0, *countPointsPerProccess;

	countPointsPerProccess = allocateIntMemory(N);

	while (currentProccess < proccessesNumber)
	{
		countPointsPerProccess[currentProccess] = N / proccessesNumber;
		if ((N % proccessesNumber)-currentProccess > 0)
		{
			countPointsPerProccess[currentProccess]++;
		}

		pointsToSend[currentProccess] = countPointsPerProccess[currentProccess] * DIMENSIONS;
		gapsArr[currentProccess] = gap;
		gap += pointsToSend[currentProccess];
		
		currentProccess++;
	}

	index = 0;
	while(index < proccessesNumber) 
	{ 
		pointGatherArr[index] = pointsToSend[index] / DIMENSIONS;
		recvCountsUpdatedPoints[index] = pointGatherArr[index] * DIMENSIONS;

		if (index == 0)
		{
			gapsGatherArr[0] = 0;
			updatedPointsGaps[0] = 0;
		}
		else
		{
			gapsGatherArr[index] = gapsGatherArr[index - 1] + pointGatherArr[index - 1];
			updatedPointsGaps[index] = recvCountsUpdatedPoints[index - 1];
		}
		index++;
	}

	numberOfPointsInProccess = pointsToSend[currentId] / DIMENSIONS;

	proccessPoints = (double**)malloc(pointsToSend[currentId] / DIMENSIONS * sizeof(double*));
	proccessVelocity = (double**)malloc(pointsToSend[currentId] / DIMENSIONS * sizeof(double*));

	index = 0;
	while(index < numberOfPointsInProccess)
	{
		if (index != 0)
		{
			proccessPoints[index] = proccessPoints[index - 1] + DIMENSIONS;
			proccessVelocity[index] = proccessVelocity[index - 1];
		}
		else
		{
			proccessPoints[0] = (double*)malloc(pointsToSend[currentId] * sizeof(double));
			proccessVelocity[0] = (double*)malloc(pointsToSend[currentId] * sizeof(double));
		}
		index++;
	}

	//Send arrays to all proccesses
	MPI_Scatterv(pointsFromFile, pointsToSend, gapsArr, MPI_DOUBLE, proccessPoints[0], pointsToSend[currentId], MPI_DOUBLE, ROOTID, MPI_COMM_WORLD);
	MPI_Scatterv(pointsVeloFromFile, pointsToSend, gapsArr, MPI_DOUBLE, proccessVelocity[0], pointsToSend[currentId], MPI_DOUBLE, ROOTID, MPI_COMM_WORLD);
	
	//every proccess write points to GPU
	
	cudaSetDevice(GPU_ZERO);
	int arraySize = numberOfPointsInProccess * DIMENSIONS * sizeof(double);
	cudaMalloc((void**)&pointsArrayInGpu, arraySize);
	cudaMalloc((void**)&velocityArrayInGpu, arraySize);
	cudaMemcpy(pointsArrayInGpu, proccessPoints[0], arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(velocityArrayInGpu, proccessVelocity[0], arraySize, cudaMemcpyHostToDevice);
	

	pointToCluster = allocateIntMemory(numberOfPointsInProccess);

	clustersArray = (double**)malloc(K * sizeof(double*));
	clustersArray[0] = (double*)malloc(numberOfPointsInProccess * DIMENSIONS * sizeof(double)); //OREL: maybe should change k to numofPoints
	index = 1;
	while(index<K)
	{
		clustersArray[index] = clustersArray[index - 1] + DIMENSIONS;
		index++;
	}


	if (currentId == ROOTID)
	{
		int clusterIndex = 0;

			while (clusterIndex < K)
			{
				int clusterPointIndex = 0;
				while (clusterPointIndex < DIMENSIONS)
				{
					int currentPointIndex = clusterPointIndex + clusterIndex * DIMENSIONS;
					clustersArray[clusterIndex][clusterPointIndex] = pointsFromFile[currentPointIndex];
					clusterPointIndex++;
				}
				clusterIndex++;
			}
	}

	MPI_Bcast(clustersArray[0], K * DIMENSIONS, MPI_DOUBLE, ROOTID, MPI_COMM_WORLD);

	qualityAtCurrentTime = QM + 1;
	timeCounter = 0;
	while (timeCounter < maxTime && qualityAtCurrentTime > QM)
	{
		movePoints(numberOfPointsInProccess, dT, proccessPoints, pointsArrayInGpu, velocityArrayInGpu);
		
		kMeansAlgorithm(MPI_COMM_WORLD, pointsArrayInGpu, numberOfPointsInProccess, K, LIMIT, pointToCluster, clustersArray, proccessPoints);
	
		//MAP POINT TO PROCCESS
		MPI_Gatherv(proccessPoints[0], numberOfPointsInProccess, MPI_DOUBLE, pointsFromFile, recvCountsUpdatedPoints,
			updatedPointsGaps, MPI_DOUBLE, ROOTID, MPI_COMM_WORLD);
		//MAP CLUSTER TO POINT
		MPI_Gatherv(pointToCluster, numberOfPointsInProccess, MPI_INT, pointClusterMapping,
			pointGatherArr, gapsGatherArr, MPI_INT, ROOTID, MPI_COMM_WORLD);



		if (currentId == ROOTID)
		{
			//Quality Measure code #OMP
			double* arrayOfClustersDiameters = calculateDiameters(pointsFromFile, N, K, pointClusterMapping);
			qualityAtCurrentTime = qualityCalculation(clustersArray, K, arrayOfClustersDiameters);

		}
		MPI_Bcast(&qualityAtCurrentTime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		if (currentId == ROOTID)
		{
			printf("Time Counter: %lf  -  Quality(dT): %lf\n", timeCounter, qualityAtCurrentTime);
			fflush(stdout);
		}

		timeCounter += dT;
	} 


	
	T = MPI_Wtime() - T;

	if (currentId == ROOTID)
	{
		FILE* outputFile = fopen(OUTPUT, "w");
		writeClusters(outputFile, clustersArray, K, T, qualityAtCurrentTime);
		printf("\ntime=%.5f\nquality=%.5f\n\n", T, qualityAtCurrentTime);
		fflush(stdout);
	}

	MPI_Finalize();
}

void initializeMpi(int argc, char **argv, int* id,int* proccessesNumber)
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, id);
	MPI_Comm_size(MPI_COMM_WORLD, proccessesNumber);
}

int* allocateIntMemory(int size) 
{
	return (int*)malloc(size * sizeof(int));
}


double* readPoints(FILE* file, int *N, int *K, int *T, double *dT, int *LIMIT, double *QM, double **velocity)
{
	double *pointsArray;
	fscanf(file, "%d %d %d %lf %d %lf\n", N, K, T, dT, LIMIT, QM);

	pointsArray = (double*)malloc((*N) * DIMENSIONS * sizeof(double));
	*velocity = (double*)malloc((*N) * DIMENSIONS * sizeof(double));

	int pointIndex = 0;
	while (pointIndex < (*N))
	{
		int cordinateIndex = 0;
		int pointsTillNow = pointIndex * DIMENSIONS;
		while (cordinateIndex < DIMENSIONS)
		{
			fscanf(file, "%lf ", &pointsArray[cordinateIndex + pointsTillNow]);
			cordinateIndex++;
		}
		cordinateIndex = 0;
		while (cordinateIndex < DIMENSIONS)
		{
			fscanf(file, "%lf ", (*velocity) + cordinateIndex + pointsTillNow);
			cordinateIndex++;
		}
		fscanf(file, "\n");
		pointIndex++;
	}

	fclose(file);
	return pointsArray;
}

void writeClusters(FILE* file, double **clustersArray, int K, double T, double QM)
{
	printf("DONE! PLEASE DO THE FOLLOWING:\n1.OPEN THE OUTPUT FILE\n2.CLOSE THE PROCCESS FROM TASK MGR (MPI BUG)");
	fflush(stdout);
	fprintf(file, "First occurrence at t = %lf with q = %.9f\n\n", T, QM);
	fprintf(file, "Centers of the clusters:\n\n");
	int currentIndex = 0;
	while (currentIndex<K)
	{
		int currentDimension = 0;
		while (currentDimension<DIMENSIONS)
		{
			fprintf(file, "%.9f ", clustersArray[currentIndex][currentDimension]);
			currentDimension++;
		}

		fprintf(file, "\n");
		currentIndex++;
	}
	fclose(file);
}


/** OMP FROM HERE  **/
double qualityCalculation(double **clustersArray, int clusterCount, double *clustersDiametersArray)
{
	int currentCluster;
	int clusterIndexPointer;
	double sharedQuality = 0;

#pragma omp parallel for private(clusterIndexPointer) reduction(+ : sharedQuality)
	for (currentCluster = 0; currentCluster < clusterCount; ++currentCluster)
	{
		clusterIndexPointer = currentCluster + 1;
		while (clusterIndexPointer<clusterCount)
		{
			int index = 0;
			double distance = 0;

			while (index<DIMENSIONS)
			{
				double a = clustersArray[currentCluster][index] - clustersArray[clusterIndexPointer][index];
				distance += a * a;
				index++;
			}
			sharedQuality += (clustersDiametersArray[currentCluster] + clustersDiametersArray[clusterIndexPointer]) / sqrt(distance);
			clusterIndexPointer++;
		}
	}

	int numberOfClusterDistances = clusterCount * (clusterCount - 1);
	return sharedQuality / numberOfClusterDistances;
}

double* calculateDiameters(double *pointsArray, int N, int K, int *pointToClusterMapping)
{
	double distance;
	double *threadForCalcDiameters;
	double *diametersArray;
	int threadCount;

	double diameter = 0.0;

	threadCount = omp_get_max_threads();

	threadForCalcDiameters = (double*)calloc(threadCount  * K, sizeof(double));

	diametersArray = (double*)malloc(K * sizeof(double));
	int currentPoint;
	int nextPointIndex;
	int ompThreadId;
	int step;
#pragma omp parallel for private(ompThreadId,nextPointIndex,step,distance) shared(threadForCalcDiameters)
	for (currentPoint = 0; currentPoint < N; ++currentPoint)
	{
		ompThreadId = omp_get_thread_num();
		step = ompThreadId * K;
		nextPointIndex = currentPoint + 1;
		while (nextPointIndex < N)
		{
			if (pointToClusterMapping[nextPointIndex] != pointToClusterMapping[currentPoint])
			{
				nextPointIndex++;
				continue;
			}
			int index = 0;
			double expoDistance = 0;

			while (index < DIMENSIONS)
			{
				double *firstPoint = pointsArray + (currentPoint * DIMENSIONS);
				double *secondPoint = pointsArray + (nextPointIndex * DIMENSIONS);
				double oneLine = (firstPoint[index] - secondPoint[index]);
				expoDistance += oneLine * oneLine;
				index++;
			}

			distance = sqrt(expoDistance);
			int currentThreadIndex = step + pointToClusterMapping[currentPoint];
			if (distance > threadForCalcDiameters[currentThreadIndex])
				threadForCalcDiameters[currentThreadIndex] = distance;
			nextPointIndex++;
		}
	}

	currentPoint = 0;
	nextPointIndex = 1;
	while (currentPoint < K)
	{
		diametersArray[currentPoint] = threadForCalcDiameters[currentPoint];
		while (nextPointIndex < threadCount)
		{
			int threadIndex = currentPoint + K * nextPointIndex;
			if (diametersArray[currentPoint] < threadForCalcDiameters[threadIndex])
				diametersArray[currentPoint] = threadForCalcDiameters[threadIndex];
			nextPointIndex++;
		}
		currentPoint++;
	}

	//	free(threadForCalcDiameters);

	return diametersArray;
}

/** OMP TILL HERE  **/

/*   KMEANS    */

void kMeansAlgorithm(MPI_Comm comm, double *gpuArrayPoints, int N, int K, int LIMIT, int *pointsToClusterMapping, double **clustersArray, double **pointsArray)
{
	int indexArr, currentClusterIndex;
	indexArr = 0;
	while (indexArr < N)
	{
		pointsToClusterMapping[indexArr] = -1;
		indexArr++;
	}

	int *gpuPointToClusterMapping;
	int *tempCluster;
	int *updatedClusterCountSizeArray;
	double  **finalCluster;

	gpuPointToClusterMapping = (int*)malloc(N * sizeof(int));
	finalCluster = (double**)malloc(K * sizeof(double*));
	finalCluster[0] = (double*)calloc(K * DIMENSIONS, sizeof(double));
	updatedClusterCountSizeArray = (int*)calloc(K, sizeof(int));
	tempCluster = (int*)calloc(K, sizeof(int));

	indexArr = 1;

	while (indexArr < K)
	{
		finalCluster[indexArr] = finalCluster[indexArr - 1] + DIMENSIONS;
		indexArr++;
	}
	
	int iteration = 0;
	int numOfPointsChangedCluster;
	while (iteration < LIMIT)
	{
		numOfPointsChangedCluster = 0;

		pointsToCluster(N, K, gpuPointToClusterMapping, gpuArrayPoints, clustersArray);

		indexArr = 0;
		while (indexArr < K)
		{

			if (gpuPointToClusterMapping[indexArr] != pointsToClusterMapping[indexArr])
			{
				numOfPointsChangedCluster++;
				pointsToClusterMapping[indexArr] = gpuPointToClusterMapping[indexArr];
			}

			currentClusterIndex = 0;
			currentClusterIndex = gpuPointToClusterMapping[indexArr];

			updatedClusterCountSizeArray[currentClusterIndex]++;

			int indextest = 0;
			while (indextest < DIMENSIONS)
			{
				finalCluster[currentClusterIndex][indextest] += pointsArray[indexArr][indextest];
				indextest++;
			}
			indexArr++;

		}

		//Combines values from all processes and distributes the result back to all processes
		int	sumOfChanges = 0;
		MPI_Allreduce(&numOfPointsChangedCluster, &sumOfChanges, 1, MPI_INT, MPI_SUM, comm);

		if (sumOfChanges == 0)
		{ //no changes
			break;
		}


		MPI_Allreduce(finalCluster[0], clustersArray[0], K * DIMENSIONS, MPI_DOUBLE, MPI_SUM, comm);

		MPI_Allreduce(updatedClusterCountSizeArray, tempCluster, K, MPI_INT, MPI_SUM, comm);

		indexArr = 0;
		while (indexArr < K)
		{
			currentClusterIndex = 0;
			while (currentClusterIndex < DIMENSIONS)
			{
				if (tempCluster[indexArr] > 1)
				{
					clustersArray[indexArr][currentClusterIndex] /= tempCluster[indexArr];
				}
				finalCluster[indexArr][currentClusterIndex] = 0;
				currentClusterIndex++;
			}
			updatedClusterCountSizeArray[indexArr] = 0;
			indexArr++;
		}

		iteration++;
	}

}