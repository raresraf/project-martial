/**
 * @76dca07c-c626-46b0-8168-5bec8ae598f2/solver_neopt.c
 * @brief Naive baseline implementation of a matrix expression solver.
 *
 * Functional Utility: Implements the matrix expression Result = (A * B * B^T) + (A^T * A)
 * using standard triple-nested iterative loops. Serves as a reference implementation 
 * for correctness and a baseline for performance profiling.
 *
 * Domain: HPC Numerical Baselines.
 */

#include "utils.h"


/**
 * @brief Computes the matrix expression via a sequence of naive multiplication stages.
 * Algorithm: Stage-based linear algebra calculation.
 * Time Complexity: $O(N^3)$.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	int i, j, k;
	// Pre-condition: Allocates zero-initialized workspace for intermediate and term buffers.
	double *M1 = (double *)calloc(N * N, sizeof(double)); 
	double *M2 = (double *)calloc(N * N, sizeof(double)); 
	double *M3 = (double *)calloc(N * N, sizeof(double)); 
	if (!M1 || !M2 || !M3)
		return NULL;

	/**
	 * Block Logic: Compute M1 = A * B.
	 * Invariant: k starts at i to exploit the upper triangular property of A.
	 */
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for(k = i; k < N; k++)
				M1[i * N + j] += A[i * N + k] * B[k * N + j];

	/**
	 * Block Logic: Compute M2 = M1 * B^T.
	 * Logic: Transposes B implicitly by swapping row and column indices in the inner loop.
	 */
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for(k = 0; k < N; k++)
			 	M2[i * N + j] += M1[i * N + k] * B[j * N + k];

	/**
	 * Block Logic: Compute M3 = A^T * A.
	 * Invariant: The loop bound `end = min(i, j)` leverages the combined sparsity 
	 * of A and its transpose.
	 */
	int end;
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			if (i < j)
				end = i;
			else
				end = j;
			for(k = 0; k <= end; k++)
			 	M3[i * N + j] += A[k * N + i] * A[k * N + j];
			}

	/**
	 * Block Logic: Final summation pass.
	 * Invariant: Aggregates the results of the two primary compute terms into M2.
	 */
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			M2[i * N + j] += M3[i * N + j];

	/**
	 * Cleanup: Deallocates temporary matrices M1 and M3.
	 */
	free(M1);
	free(M3);
	return M2;
}
