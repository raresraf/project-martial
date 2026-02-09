/**
 * @file solver_neopt.c
 * @brief Semantic documentation for solver_neopt.c. 
 *        This is a placeholder. Detailed semantic analysis will be applied later.
 */

#include "utils.h"



double *getTranspose(int N, double *A)
{
	int i, j;
	double *trA = (double *)calloc(N * N, sizeof(double));
	for ( i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			trA[j * N + i] = A[i * N + j];
		}
	}
	return trA;
}
double *getTransposeA(int N, double *A)
{
	int i, j;
	double *trA = (double *)calloc(N * N, sizeof(double));
	for ( i = 0; i < N; i++)
	{
		for ( j = i; j < N; j++)
		{
			trA[j * N + i] = A[i * N + j];
		}
	}
	return trA;
}

double *normalMul(int N, double *A, double *B)
{
	int i, j, k;
	double *C = (double *)calloc(N * N, sizeof(double));
	for ( i = 0; i < N; i++)
	{
		for ( j = 0; j < N; j++)
		{

			for ( k = 0; k < N; k++)
			{
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
	return C;
}

double *upperMul(int N, double *A, double *B)
{
	int i, j, k;
	double *C = (double *)calloc(N * N, sizeof(double));
	for ( i = 0; i < N; i++)
	{
		for ( j = 0; j < N; j++)
		{
			for ( k = i; k < N; k++)
			{
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
	return C;
}
double *lowerMul(int N, double *A, double *B)
{
	int i, j, k;
	double *C = (double *)calloc(N * N, sizeof(double));

	for ( i = 0; i < N; i++)
	{
		for ( j = 0; j < N; j++)
		{
			for ( k = 0; k <= i; k++)
			{
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
	return C;
}
double *my_solver(int N, double *A, double *B)
{
	int i, j;
	double *C = (double *)malloc(N * N * sizeof(double));
	double *AtA;
	double *BBt;
	double *ABBt;

	double *trA = getTransposeA(N, A);
	double *trB = getTranspose(N, B);

	AtA = lowerMul(N, trA, A);

	BBt = normalMul(N, B, trB);
	ABBt = upperMul(N, A, BBt);
	for ( i = 0; i < N; i++)
	{
		for ( j = 0; j < N; j++)
		{
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}
	free(AtA);
	free(BBt);
	free(ABBt);
	free(trA);
	free(trB);
	printf("NEOPT SOLVER\n");
	return C;
}
