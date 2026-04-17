/**
 * @file solver_opt.c
 * @brief Manually optimized matrix solver implementation.
 * Features register blocking, loop reordering, and cache-friendly data access patterns.
 */

#include "utils.h"



double *getTranspose(int N, double *A)
{
	register int i, j;
	double *trA = (double *)calloc(N * N, sizeof(double));
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i++)
	{
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
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
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i++)
	{
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
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

	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (j = 0; j < N; j++)
	{
		double *pb_origin = &B[j]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (i = 0; i < N; i++)
		{
			double *pa = &A[i * N]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
			double *pb = pb_origin; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0;
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (k = 0; k < N; k++)
			{
				sum += *pa * *pb; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
				pa++;
				pb += N; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
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
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (j = 0; j < N; j++)
	{
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (i = 0; i < N; i++)
		{

			register double sum = 0;

			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
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

	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (j = 0; j < N; j++)
	{
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (i = 0; i < N; i++)
		{

			register double sum = 0;
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
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
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i++)
	{
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
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
