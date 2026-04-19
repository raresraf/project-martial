/**
 * @raw/ab6ed51d-ae33-48f2-bee3-efa7e7895107/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver computing C = (A * B) * B^T + A^T * A.
 * Algorithm: Implementation utilizing basic standard loops sequentially without loop bounds checks or transpositions.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ due to intermediary multi-buffer usage.
 */

#include "utils.h"
#include <string.h>

double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	int i,j,k;
	
	/**
	 * Functional Utility: Initializes heap-allocated structures mapping to array partitions handling separate outputs.
	 */
	double *C = malloc(sizeof(double) * N * N);
	if (!C) 
		return NULL;


	double *AB = malloc(sizeof(double) * N * N);
	if (!AB)
		return NULL;

	double *BBt = malloc(sizeof(double) * N * N);
	if (!BBt)
		return NULL;

	double *AtA = malloc(sizeof(double) * N * N);
	if (!AtA)
		return NULL;

	/**
	 * Block Logic: Computes the base product AB = A * B.
	 * Invariant: Operates independently of specific matrix bounds traversing the full NxN block structure.
	 */
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			AB[i * N + j] = 0.0;
			for (k = 0; k < N; k++)
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
		}
	
	/**
	 * Block Logic: Evaluates trailing product AB * B^T defining the vector output inside BBt.
	 * Invariant: Extracts parameters directly reflecting matrix B transposition sequentially mapping indices.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			BBt[i * N + j] = 0.0;
			for (k = 0; k < N; k++) {
				BBt[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Block Logic: Derives A^T * A internally setting the solution inside AtA.
	 * Invariant: Flips initial input arrays executing element-wise calculations.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			AtA[i * N + j] = 0.0;
			for (k = 0; k < N; k++) {
				AtA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Performs the final reduction translating the formula into C.
	 */
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			C[i * N + j] = BBt[i * N + j] + AtA[i * N + j];

	free(BBt);
	free(AtA);
	free(AB);
	return C;
}
