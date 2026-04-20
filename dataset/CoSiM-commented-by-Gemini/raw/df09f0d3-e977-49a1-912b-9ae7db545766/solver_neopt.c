/**
 * @file solver_neopt.c
 * @brief Unoptimized standard implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Naive multi-loop dense matrix multiplication with upper triangular conditions.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store the intermediate matrices.
 */

#include "utils.h"
#include <string.h>


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	int i, j, k;
	double *At = (double*) calloc(N * N, sizeof(double));
	double *Bt = (double*) calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(i = 0; i < N; i++) {
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for(j = 0; j < N; j++) {
            At[j * N + i] = A[i * N + j];
        }
    }
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(i = 0; i < N; i++) {
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for(j = 0; j < N; j++) {
            Bt[j * N + i] = B[i * N + j];
        }
    }

	
	double *C1 = (double*) calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (j = 0; j < N; j++) {
				/**
				 * Block Logic: Iterative processing loop.
				 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
				 */
				for (k = i; k < N; k++) { 
						C1[i * N + j] += A[i * N + k] * B[k * N + j];
				}
			}
		}
	
	double *C2 = (double*) calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (j = 0; j < N; j++) {
				/**
				 * Block Logic: Iterative processing loop.
				 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
				 */
				for (k = 0; k < N; k++) { 
						C2[i * N + j] += C1[i * N + k] * Bt[k * N + j];
				}
			}
		}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (j = 0; j < N; j++) {
				/**
				 * Block Logic: Iterative processing loop.
				 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
				 */
				for (k = 0; k <= i && k <= j; k++) { 
						C2[i * N + j] += At[i * N + k] * A[k * N + j];
				}
			}
		}

	free(At);
	free(Bt);
	free(C1);

	return C2;
}
