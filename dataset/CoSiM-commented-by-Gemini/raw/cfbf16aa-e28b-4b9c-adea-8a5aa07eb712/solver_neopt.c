/**
 * @file solver_neopt.c
 * @brief Unoptimized standard implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Naive multi-loop dense matrix multiplication with upper triangular conditions.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store the intermediate matrices.
 */

#include "utils.h"

#define MIN(a, b) ((a < b)?(a):(b))


double *sum(int N, double *X, double *Y) {
	int i = 0, j = 0;
	double *sum_result = malloc(N * N * sizeof(double));

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
			sum_result[i * N + j] = X[i * N + j] + Y[i * N + j];
		}
	}
	return sum_result;
}


double *mul_B_BT(int N, double *B) {
	int i = 0, j = 0, k = 0;
	double *mul_result = calloc(N * N, sizeof(double));

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
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for(k = 0; k < N; k++) {
				mul_result[i * N + j] += B[i * N + k] * B[j * N + k];
			}
		}
	}
	return mul_result;
}


double *mul_A_MAT(int N, double *A, double *MAT) {
	int i = 0, j = 0, k = 0;
	double *mul_result = calloc(N * N, sizeof(double));

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
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for(k = i; k < N; k++) {
				mul_result[i * N + j] += A[i * N + k] * MAT[k * N + j];
			}
		}
	}
	return mul_result;
}


double *mul_AT_A(int N, double *A) {
	int i = 0, j = 0, k = 0;
	double *mul_result = calloc(N * N, sizeof(double));

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
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for(k = 0; k <= MIN(i, j); k++) {
				
				mul_result[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}
	return mul_result;
}

/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	double *B_X_BT = mul_B_BT(N, B);
	double *A_X_B_X_BT = mul_A_MAT(N, A, B_X_BT);
	double *AT_X_A = mul_AT_A(N, A);
	double *result = sum(N, A_X_B_X_BT, AT_X_A);
	free(B_X_BT);
	free(A_X_B_X_BT);
	free(AT_X_A);
	return result;
}
