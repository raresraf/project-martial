

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "utils.h"
#include "cblas.h"

void allocate_matrices(int N, double **C, double **AB, double **ABBt, double **AtA)
{
	*C = malloc(N * N * sizeof(**C));
	if (NULL == *C)
		exit(EXIT_FAILURE);

	*AB = malloc(N * N * sizeof(**AB));
	if (NULL == *AB)
		exit(EXIT_FAILURE);
   
	*ABBt = malloc(N * N * sizeof(**ABBt));
	if (NULL == *ABBt)
		exit(EXIT_FAILURE);

	*AtA = malloc(N * N * sizeof(**AtA));
	if (NULL == *AtA)
		exit(EXIT_FAILURE);
}

double* my_solver(int N, double *A, double *B) 
{
	double *C;
	double *AB;
	double *ABBt;
	double *AtA;
	int i, j;

	allocate_matrices(N, &C, &AB, &ABBt, &AtA);

	
	memcpy(AtA, A, N * N * sizeof(*AtA));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		AtA, N
	);

	
	memcpy(AB, B, N * N * sizeof(*AtA));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		AB, N
	);

	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1.0,
		AB,
		N,
		B,
		N,
		0.0,
		ABBt,
		N
	);

	for (i = 0; i < N; i++){
        	for(j = 0; j < N; j++) 
            		C[i * N + j] += ABBt[i * N + j] + AtA[i * N + j];
    	}
	free(AB);
	free(ABBt);
	free(AtA);
	return C;
}
