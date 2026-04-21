/**
 * @file solver_neopt.c
 * @brief Naive reference implementation of the matrix expression solver.
 *
 * Computes: Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 *
 * Algorithm: Series of standard triple-nested loop matrix multiplications.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store intermediate product matrices.
 *
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

#include "utils.h"

/**
 * my_solver - Naive implementation using standard C nested loops.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	int k, j, i;

	/**
	 * Pre-condition: Allocation and zero-initialization of result and 
	 * scratch buffers.
	 */
	double *C = (double*) calloc(N * N, sizeof(double));
    if (C == NULL) {
		printf("Calloc failed!\n");
      	exit(1);
	}
    double *D = (double*) calloc(N * N, sizeof(double));
    if (D == NULL) {
		printf("Calloc failed!\n");
      	exit(1);
	}

    /**
     * Block Logic: Compute C = A * B.
     * Optimization: Exploits upper triangularity of A by starting k from i.
     */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = i; k < N; k++) {
                *(C + i * N + j) += *(A + i * N + k) * *(B + k * N + j);
            }
        }
    }
    
    /**
     * Block Logic: Compute D = C * B^T.
     * Invariant: Accesses B[j][k] as B^T[k][j] to avoid explicit transposition.
     */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                *(D + i * N + j) += *(C + i * N + k) * *(B + j * N + k);
            }
        }
    }
    
	/**
	 * Block Logic: Accumulate D += A^T * A.
	 * Algorithm: Direct dot product implementation of the Gramian matrix.
	 */
	for (k = 0; k < N; k++) {
		for (i = k; i < N; i++) {
            for (j = k; j < N; j++) {
                *(D + i * N + j) += *(A + k * N + i) * *(A + k * N + j);
            }
		}
	}
	free(C);
	return D;
}
