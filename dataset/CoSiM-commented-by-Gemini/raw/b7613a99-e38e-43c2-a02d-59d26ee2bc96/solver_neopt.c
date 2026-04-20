/**
 * @file solver_neopt.c
 * @brief Unoptimized standard implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Naive multi-loop dense matrix multiplication with upper triangular conditions.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store the intermediate matrices.
 */


#include <stdlib.h>
#include <string.h>
#include "utils.h"


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {

	double *C;
	double *AA;
	double *AB;

	
	C = calloc(N * N, sizeof(*C));
	AA = calloc(N * N, sizeof(*AA));
	AB = calloc(N * N, sizeof(*AB));

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (int i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (int j = 0; j < N; j++) {
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (int k = 0; k < j + 1; k++) {
				AA[i  * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (int i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (int j = 0; j < N; j++) {
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (int k = i; k < N; k++) {
				AB[i  * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (int i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (int j = 0; j < N; j++) {
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (int k = 0; k < N; k++) {
				C[i  * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (int i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (int j = 0; j < N; j++) {
			C[i * N + j] += AA[i * N + j];
		}
	}


	printf("NEOPT SOLVER\n");
	
	free(AA);
	free(AB);

	return C;
}
