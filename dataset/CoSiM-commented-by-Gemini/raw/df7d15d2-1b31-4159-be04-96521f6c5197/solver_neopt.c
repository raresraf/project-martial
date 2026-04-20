/**
 * @file solver_neopt.c
 * @brief Unoptimized standard implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Naive multi-loop dense matrix multiplication with upper triangular conditions.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store the intermediate matrices.
 */

#include "utils.h"
#include <string.h>

void allocate_memory(int N, double **C, double **A_t, double **B_t,
					double **left_side, double **right_side, double **tmp)
{
	*C = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (!(*C)) {
		exit(-1);
	}

	*A_t = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (!(*A_t)) {
		exit(-1);
	}

	*B_t = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (!(*B_t)) {
		exit(-1);
	}

	*left_side = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (!(*left_side)) {
		exit(-1);
	}

	*right_side = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (!(*right_side)) {
		exit(-1);
	}

	*tmp = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (!(*tmp)) {
		exit(-1);
	}
}


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	
	size_t i, j, k;
	double *C, *A_t, *B_t, *left_side, *right_side, *tmp;

	allocate_memory(N, &C, &A_t, &B_t, &left_side, &right_side, &tmp);

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			*(A_t + N * i + j) = *(A + N * j + i);
			*(B_t + N * i + j) = *(B + N * j + i);
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			*(left_side + N * i + j) = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; ++k) {
				
				*(left_side + N * i + j) += *(A + N * i + k) * *(B + N * k + j);
			}
		}
	}

	
	memcpy(tmp, left_side, N * N * sizeof(double));

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			*(left_side + N * i + j) = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; ++k) {
				*(left_side + N * i + j) += *(tmp + N * i + k) * *(B_t + N * k + j);
			}
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			*(right_side + N * i + j) = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < j + 1; ++k) {
				*(right_side + N * i + j) += *(A_t + N * i + k) * *(A + N * k + j);
			}
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			*(C + N * i + j) = *(left_side + N * i + j) + *(right_side + N * i + j);
		}
	}

	free(left_side);
	free(right_side);
	free(tmp);
	free(A_t);
	free(B_t);

	return C;
}
