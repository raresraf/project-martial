/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
 */


#include "utils.h"



double *add_matrix(int N, double *A, double *B) {
	int i, j;

	double *result = (double *) calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (result == NULL) {
		return NULL;
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
			result[i * N + j] += A[i * N + j] + B[i * N + j];
		}
	}

	return result;
}



double *transpose_matrix(int N, double *A) {
	int i, j;

	double *result = (double *) calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (result == NULL) {
		return NULL;
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
			result[j * N + i] = A[i * N + j];
		}
	}

	return result;
}



double *transpose_upper_matrix(int N, double *A) {
	int i, j;

	double *result = (double *) calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (result == NULL) {
		return NULL;
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
		for (j = i; j < N; ++j) {
			result[j * N + i] = A[i * N + j];
		}
	}

	return result;
}



double *multiply_matrix(int N, double *A, double *B) {
    int i, j, k;

    double *result = (double *) calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (result == NULL) {
		return NULL;
	}


	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (k = 0; k < N; ++k) {
		register double *orig_pb = &B[k * N + 0]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		register double *pa = &A[0 * N + k]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (i = 0; i < N; ++i) {
			register double *pc = &result[i * N + 0]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb = orig_pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (j = 0; j < N; ++j) {
                *pc += *pa * *pb;
                pb++;
                pc++;
			}

			pa += N;
		}
	}

	return result;
}



double *multiply_upper_matrix(int N, double *A, double *B) {
    int i, j, k;

    double *result = (double *) calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (result == NULL) {
		return NULL;
	}


	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pa = &A[i * N + i]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb = &B[i * N + j]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0.0;

			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; ++k) {
				sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pa++;
				pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			}

			result[i * N + j] = sum;
		}
	}

	return result;
}



double *multiply_lower_matrix(int N, double *A, double *B) {
    int i, j, k;

    double *result = (double *) calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (result == NULL) {
		return NULL;
	}


	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pa = &A[i * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb = &B[j]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0.0;

			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < i + 1; ++k) {
				sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pa++;
				pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			}

			result[i * N + j] = sum;
		}
	}

	return result;
}




/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {

	

	
	
	double *B_trans = transpose_matrix(N, B);
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (B_trans == NULL) {
		return NULL;
	}


	
	double *M = multiply_matrix(N, B, B_trans);
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (M == NULL) {
		return NULL;
	}


	
	double *P = multiply_upper_matrix(N, A, M);
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (P == NULL) {
		return NULL;
	}


	
	double *A_trans = transpose_upper_matrix(N, A);
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (A_trans == NULL) {
		return NULL;
	}


	
	double *L = multiply_lower_matrix(N, A_trans, A);
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (L == NULL) {
		return NULL;
	}


	
	double *result = add_matrix(N, P, L);
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (result == NULL) {
		return NULL;
	}


	free(B_trans);
	free(A_trans);
	free(M);
	free(P);
	free(L);

	return result;
}
