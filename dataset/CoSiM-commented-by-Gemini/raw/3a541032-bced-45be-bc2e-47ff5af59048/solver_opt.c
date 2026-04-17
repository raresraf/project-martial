/**
 * @file solver_opt.c
 * @brief Optimized implementation of matrix solver utilizing register caching.
 *
 * Implements C = A * B * B^T + A^T * A.
 * Optimizations include pre-calculating indices and using register variables
 * for fast access within inner loops.
 */

#include "utils.h"

/**
 * @brief Solves the matrix equation A * B * B^T + A^T * A with loop optimizations.
 *
 * @param N Matrix dimension.
 * @param A Pointer to the first input matrix (upper triangular).
 * @param B Pointer to the second input matrix.
 * @return Pointer to the resulting matrix.
 */
double* my_solver(int N, double *A, double* B) {
	
	register int i = 0;
	register int j = 0;
	register int k = 0;

	register int size = N * N * sizeof(double);

	double *result_AB = malloc(size);
	

	/**
	 * @brief Computes A * B, considering A is upper triangular.
	 * Pre-condition: Memory allocated for partial result.
	 * Invariant: Row i of result_AB is computed.
	 */
	for (i = 0; i < N; i++) {
		register int indx = i * N;
		/**
		 * @brief Traverse columns of B.
		 * Pre-condition: Base row index indx is calculated.
		 * Invariant: sum accumulates the product for result_AB[i, j].
		 */
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			/**
			 * @brief Compute dot product utilizing A's upper triangular form.
			 * Pre-condition: Starts at diagonal element k=i.
			 * Invariant: sum maintains running total.
			 */
			for (k = i; k < N; k++) {
				sum += A[indx + k] * B[k * N + j];
			}
			result_AB[indx + j] = sum;
		}
	}

	
	double *C = malloc(size);

	/**
	 * @brief Computes C = (A * B) * B^T + A^T * A simultaneously.
	 * Pre-condition: result_AB is fully computed.
	 * Invariant: Row i of C is computed.
	 */
	for (i = 0; i < N; i++) {
		register int indx1 = i * N;
		/**
		 * @brief Traverse columns.
		 * Pre-condition: Base row index indx1 is calculated.
		 * Invariant: sum and sumA accumulate dot products for respective matrices.
		 */
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			register double sumA = 0.0;
			register int jn = j * N;
			/**
			 * @brief Compute both dot products in a single inner loop to maximize data reuse.
			 * Pre-condition: Base indices jn and indx1 are calculated.
			 * Invariant: Running totals are kept in registers.
			 */
			for (k = 0; k < N; k++) {
				register int kn = k * N;
				sum += result_AB[indx1 + k] * B[jn + k];
				sumA += A[kn + i] * A[kn + j];
			}
			C[indx1 + j] = sum + sumA;
		}
	}

	printf("OPT SOLVER\n");
	free(result_AB);
	return C;	
}
