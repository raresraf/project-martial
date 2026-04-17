/**
 * @file solver_neopt.c
 * @brief Unoptimized implementation of matrix solver.
 *
 * Implements the mathematical equation C = A * B * B^T + A^T * A.
 * Computes partial results into separate matrices before final addition.
 */

#include "utils.h"

/**
 * @brief Solves the matrix equation A * B * B^T + A^T * A sequentially.
 *
 * @param N Matrix dimension.
 * @param A Pointer to the first input matrix.
 * @param B Pointer to the second input matrix.
 * @return Pointer to the resulting matrix.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	int i = 0;
	int j = 0;
	int k = 0;

	
	double *result_A = calloc(N * N, sizeof(double));
	
	/**
	 * @brief Computes A^T * A.
	 * Pre-condition: Memory for result_A is allocated.
	 * Invariant: Elements up to result_A[i, j] are computed.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * @brief Iterate over columns.
		 * Pre-condition: Valid row index i.
		 * Invariant: Computes dot product.
		 */
		for (j = 0; j < N; j++) {
			result_A[i * N + j] = 0;
			/**
			 * @brief Inner loop for dot product accumulation.
			 * Pre-condition: Starts with 0.
			 * Invariant: Accumulates multiplication.
			 */
			for (k = 0; k < N; k++) {
				result_A[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	double *result_AB = calloc(N * N, sizeof(double));
	

	
	/**
	 * @brief Computes A * B, considering A is upper triangular.
	 * Pre-condition: Memory for result_AB is allocated.
	 * Invariant: Elements up to result_AB[i, j] are computed.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * @brief Iterate over columns.
		 * Pre-condition: Valid row index i.
		 * Invariant: Computes dot product.
		 */
		for (j = 0; j < N; j++) {
			result_AB[i * N + j] = 0;
			/**
			 * @brief Inner loop leveraging upper triangular property (k starts from i).
			 * Pre-condition: Starts with 0.
			 * Invariant: Accumulates multiplication.
			 */
			for (k = i; k < N; k++) {
				result_AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	double *result_ABBt = calloc(N * N, sizeof(double));

	/**
	 * @brief Computes (A * B) * B^T.
	 * Pre-condition: result_AB contains the A * B product.
	 * Invariant: Elements up to result_ABBt[i, j] are computed.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * @brief Iterate over columns.
		 * Pre-condition: Valid row index i.
		 * Invariant: Computes dot product.
		 */
		for (j = 0; j < N; j++) {
			result_ABBt[i * N + j] = 0;
			/**
			 * @brief Inner loop multiplying by transpose implicitly.
			 * Pre-condition: Starts with 0.
			 * Invariant: Accumulates multiplication.
			 */
			for (k = 0; k < N; k++) {
				result_ABBt[i * N + j] += result_AB[i * N + k] * B[j * N + k];
			}
		}
	}

	double *C = calloc(N * N, sizeof(double));


	/**
	 * @brief Adds the two partial results together.
	 * Pre-condition: Both partial matrices are computed.
	 * Invariant: Matrix C is formed iteratively.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * @brief Element-wise addition.
		 * Pre-condition: Valid row index i.
		 * Invariant: Elements up to j are summed.
		 */
		for (j = 0 ; j < N; j++) {
			C[i * N + j] = result_ABBt[i * N + j] + result_A[i * N + j];
		}
	}
	free(result_A);
	free(result_AB);
	free(result_ABBt);
	return C;

}
