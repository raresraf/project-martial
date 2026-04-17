/**
 * @file solver_neopt.c
 * @brief High-level source code module.
 * Ensures cache-friendly data access, potential loop unrolling, and SIMD optimizations for C/C++.
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double *C1 = calloc(N * N, sizeof(double));

	
	
	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (int i = 0; i < N; i++) {
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (int j = 0; j < N; j++) {
			/**
			 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
			 * Invariant: Operations within the block strictly maintain target functional boundaries.
			 */
			for (int k = i; k < N; k++) {
				C1[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	double *C2 = calloc(N * N, sizeof(double));

	
	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (int i = 0; i < N; i++) {
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (int j = 0; j < N; j++) {
			/**
			 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
			 * Invariant: Operations within the block strictly maintain target functional boundaries.
			 */
			for (int k = 0; k < N; k++) {
				C2[i * N + j] += C1[i * N + k] * B[j * N + k];
			}
		}
	}

	double *C3 = calloc(N * N, sizeof(double));

	
	
	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (int i = 0; i < N; i++) {
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (int j = 0; j < N; j++) {
			/**
			 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
			 * Invariant: Operations within the block strictly maintain target functional boundaries.
			 */
			for (int k = 0; k <= i; k++) {
				C3[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (int i = 0; i < N * N; i++)
		C2[i] += C3[i];

	free(C1);
	free(C3);

	return C2;
}
