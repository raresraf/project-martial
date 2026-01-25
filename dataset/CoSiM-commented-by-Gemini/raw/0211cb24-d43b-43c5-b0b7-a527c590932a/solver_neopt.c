/**
 * @file solver_neopt.c
 * @brief Naive implementation of a matrix solver.
 *
 * This file contains a straightforward, non-optimized implementation of the
 * `my_solver` function. It performs matrix transposition and multiplication
 * using nested loops, which is simple to understand but generally inefficient
 * for large matrices.
 */
#include "utils.h"
#include <stdlib.h>

/**
 * @brief Solves a matrix equation using a naive, non-optimized approach.
 * @param N The dimension of the matrices.
 * @param A A pointer to the first input matrix.
 * @param B A pointer to the second input matrix.
 * @return A pointer to the resulting matrix.
 *
 * This function computes the matrix expression `A * B * B' + A' * A`, where `A'`
 * and `B'` are the transposes of matrices `A` and `B`, respectively. The
 * computation is performed using explicit nested loops for all matrix
 * operations.
 */
double* my_solver(int N, double *A, double *B) {
	int i, j, k;

	printf("NEOPT SOLVER\n");

	double *at = calloc(N * N, sizeof(double));
	if (at == NULL)
		exit(EXIT_FAILURE);

	double *bt = calloc(N * N, sizeof(double));
	if (bt == NULL)
		exit(EXIT_FAILURE);

	double *res1 = calloc(N * N, sizeof(double));
	if (res1 == NULL)
		exit(EXIT_FAILURE);

	double *res2 = calloc(N * N, sizeof(double));
	if (res2 == NULL)
		exit(EXIT_FAILURE);

	double *res3 = calloc(N * N, sizeof(double));
        if (res3 == NULL)
                exit(EXIT_FAILURE);

	double *res = calloc(N * N, sizeof(double));
        if (res == NULL)
                exit(EXIT_FAILURE);


	
	// Computes the transpose of matrices A and B.
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			at[j * N + i] = A[i *  N + j];
			bt[j * N + i] = B[i *  N + j];
		}

	
	// Computes the matrix multiplication res1 = A * B.
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = i; k < N; k++) {
				res1[i * N + j] += A[i * N + k]
					* B[k * N + j];
			}

	
	// Computes the matrix multiplication res2 = res1 * B'.
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++) {
				res2[i * N + j] += res1[i * N + k]
					* bt[k * N + j];
			}

	
	// Computes the matrix multiplication res3 = A' * A.
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k <= j; k++) {
				res3[i * N + j] += at[i * N + k]
					* A[k * N + j];
			}

	
	// Computes the final result res = res2 + res3.
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			res[i * N + j] = res2[i * N + j] + res3[i * N + j];
		}

	free(at);
	free(bt);
	free(res1);
	free(res2);
	free(res3);
	return res;
}
