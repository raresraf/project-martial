/**
 * @file solver_neopt.c
 * @brief Naive reference implementation of the matrix expression solver.
 *
 * Computes: Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 *
 * Algorithm: Standard nested-loop matrix multiplications.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store multiple intermediate matrices.
 *
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

#include "utils.h"

/**
 * my_solver - Naive implementation using standard C nested loops.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	/**
	 * Pre-condition: Allocation and zero-initialization of result and 
	 * scratch buffers.
	 */
	double *AB = malloc(N * N * sizeof(double));
	double *ABBt = malloc(N * N * sizeof(double));
	double *AtA = malloc(N * N * sizeof(double));
	double *C = malloc(N * N * sizeof(double));

	if (AB == NULL || ABBt == NULL || AtA == NULL || C == NULL){
		fprintf(stderr, "malloc error\n");
		exit(EXIT_FAILURE);
	}

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Exploits upper triangularity of A by starting k from i.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			AB[i * N + j] = 0;
			for (int k = 0; k < N; k++) {
				if (i <= k)
					AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Compute ABBt = AB * B^T.
	 * Invariant: B^T is implicitly accessed via index swapping to avoid 
	 * explicit transposition.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			ABBt[i * N + j] = 0;
			for (int k = 0; k < N; k++) {
				ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Block Logic: Compute AtA = A^T * A.
	 * Optimization: Exploits sparse structure of upper triangular matrix A
	 * in the symmetric product calculation.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			AtA[i * N + j] = 0;
			for (int k = 0; k < N; k++) {
				if (i >= k && k <= j)
					AtA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Final summation pass.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
