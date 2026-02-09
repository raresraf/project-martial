/**
 * @file solver_opt.c
 * @brief Semantic documentation for solver_opt.c. 
 *        This is a placeholder. Detailed semantic analysis will be applied later.
 */

#include "utils.h"



double *getTranspose(int N, double *A)
{
	register int i, j;
	double *trA = (double *)calloc(N * N, sizeof(double));
	for (i = 0; i < N; i++)
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
	register int i, j;
	double *trA = (double *)calloc(N * N, sizeof(double));
	for (i = 0; i < N; i++)
	{
		for (j = i; j < N; j++)
		{
			trA[j * N + i] = A[i * N + j];
		}
	}
	return trA;
}

double *normalMul(int N, double *A, double *B)
{
	register int i, j, k;
	double *C = (double *)calloc(N * N, sizeof(double));

	for (j = 0; j < N; j++)
	{
		double *pb_origin = &B[j];
		for (i = 0; i < N; i++)
		{
			double *pa = &A[i * N];
			double *pb = pb_origin;
			register double sum = 0;
			for (k = 0; k < N; k++)
			{
				sum += *pa * *pb;
				pa++;
				pb += N;
			}
			C[i * N + j] = sum;
		}
	}
	return C;
}

double *upperMul(int N, double *A, double *B)
{
	register int i, j, k;
	double *C = (double *)calloc(N * N, sizeof(double));
	for (j = 0; j < N; j++)
	{
		for (i = 0; i < N; i++)
		{

			register double sum = 0;

			for (k = i; k < N; k++)
			{
				sum += A[i * N + k] * B[k * N + j];
			}
			C[i * N + j] = sum;
		}
	}
	return C;
}
double *lowerMul(int N, double *A, double *B)
{
	register int i, j, k;
	double *C = (double *)calloc(N * N, sizeof(double));

	for (j = 0; j < N; j++)
	{
		for (i = 0; i < N; i++)
		{

			register double sum = 0;
			for (k = 0; k <= i; k++)
			{

				sum += A[i * N + k] * B[k * N + j];
			}
			C[i * N + j] = sum;
		}
	}
	return C;
}
double *my_solver(int N, double *A, double *B)
{
	register int i, j;
	double *C = (double *)calloc(N * N, sizeof(double));

	double *trA = getTransposeA(N, A);
	double *trB = getTranspose(N, B);

	double *AtA = lowerMul(N, trA, A);

	double *BBt = normalMul(N, B, trB);
	double *ABBt = upperMul(N, A, BBt);
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{

			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}
	free(AtA);
	free(BBt);
	free(ABBt);
	free(trA);
	free(trB);
	printf("OPT SOLVER\n");
	return C;
}
