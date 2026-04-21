/**
 * @662cf250-db14-4761-b30d-678fa03cf1ab/solver_neopt.c
 * @brief Baseline implementation of a matrix expression solver.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using basic 
 * iterative triple-nested loops.
 * Domain: HPC Numerical Baselines.
 */

#include "utils.h"
#include <string.h>
#include <stdlib.h>


/**
 * @brief Naive matrix solver kernel.
 * Logic: Implements the target matrix expression using standard row-column dot products.
 */
double* my_solver(int N, double *A, double* B) {
	
	double *AB = (double*)malloc(N * N * sizeof(double));
	double *ABBt = (double*)malloc(N * N * sizeof(double));
	double *AtA = (double*)malloc(N * N * sizeof(double));
	double *Res = (double*)malloc(N * N * sizeof(double));
	int i, j, k;

	/**
	 * Block Logic: Compute AB = A * B.
	 * Invariant: Exploits A's upper triangular sparsity (k starts at i).
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			AB[i * N + j] = 0;
			for (k = i; k < N; k++) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Compute ABBt = AB * B^T.
	 * Logic: Accesses B with indices swapped to simulate transposition.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			ABBt[i * N + j] = 0;
			for (k = 0; k < N; k++) {
				ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Block Logic: Compute AtA = A^T * A.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			AtA[i * N + j] = 0;
			for (k = 0; k < N; k++) {
				AtA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Result aggregation.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			Res[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);

	return Res;
}
