/**
 * @78967c0c-ba90-43f5-87be-b35b30882925/solver_neopt.c
 * @brief Naive implementation of a matrix expression solver.
 *
 * Functional Utility: Implements the matrix expression Result = (A * B) * B^T + (A^T * A)
 * using standard triple-nested iterative loops. This version serves as a correctness 
 * reference and performance baseline.
 *
 * Domain: HPC Numerical Baselines.
 */

#include "utils.h"


/**
 * @brief Computes the matrix expression via sequence of naive stages.
 * Algorithm: Stage-based linear algebra calculation.
 * Time Complexity: $O(N^3)$.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	
	double *At;
	double *Bt;
	double *C;
	
	double *Aux1;
	double *Aux2;
	
	int i, j, k;
	
	// Pre-condition: Allocates heap workspace for all intermediate and term buffers.
	At = calloc(N * N, sizeof(double));
	Bt = calloc(N * N, sizeof(double));
	C = calloc(N * N, sizeof(double));
	
	Aux1 = calloc(N * N, sizeof(double));
	Aux2 = calloc(N * N, sizeof(double));
	
	/**
	 * Block Logic: Compute A^T.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			At[j * N + i] = A[i * N + j];
		}
	}
	
	/**
	 * Block Logic: Compute B^T.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			Bt[j * N + i] = B[i * N + j];
		}
	}
	
	/**
	 * Block Logic: Compute Stage 1: Aux1 = A * B.
	 * Invariant: k starts at i to exploit upper triangularity of A.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				Aux1[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
	
	/**
	 * Block Logic: Compute Stage 2: Aux2 = Aux1 * B^T.
	 * Logic: Leverages the explicitly computed `Bt` for row-major dot product.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				Aux2[i * N + j] += Aux1[i * N + k] * Bt[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Compute Stage 3: Aux1 = A^T * A.
	 * Invariant: Exploits the combined sparsity of upper triangular A and its transpose.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			Aux1[i * N + j] = 0;

			for (k = 0; k <= i; k++) {
				Aux1[i * N + j] += At[i * N + k] * A[k * N + j];
			}
		}
	}
	
	/**
	 * Block Logic: Final summation pass.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = Aux1[i * N + j] + Aux2[i * N + j];
		}
	}
	
	/**
	 * Cleanup: Reclaims all temporary matrix buffers.
	 */
	free(At);
	free(Bt);
	
	free(Aux1);
	free(Aux2);
	
	return C;
}
