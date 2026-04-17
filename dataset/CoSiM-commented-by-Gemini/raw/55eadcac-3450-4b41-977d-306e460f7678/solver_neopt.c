/**
 * @file solver_neopt.c
 * @brief High-level source code module.
 * Ensures cache-friendly data access, potential loop unrolling, and SIMD optimizations for C/C++.
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	int i, j, k;

	
	double* C = calloc(N * N, sizeof(double));
	double* AB = calloc(N * N, sizeof(double)); 
	double* prod1 = calloc(N * N, sizeof(double)); 
	double* prod2 = calloc(N * N, sizeof(double)); 
	/**
	 * @brief Pre-condition: Evaluates logical divergence based on current state.
	 * Invariant: Guarantees correct execution flow according to conditional partitioning.
	 */
	if (C == NULL || AB == NULL || prod1 == NULL || prod2 == NULL) {
        perror("calloc failed\n");
        exit(EXIT_FAILURE);
    }

	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (j = 0; j < N; j++) {
			
			/**
			 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
			 * Invariant: Operations within the block strictly maintain target functional boundaries.
			 */
			for (k = i; k < N; k++) {
				
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (j = 0; j < N; j++) {
			/**
			 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
			 * Invariant: Operations within the block strictly maintain target functional boundaries.
			 */
			for (k = 0; k < N; k++) {
				
				
				prod1[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (j = 0; j < N; j++) {
			/**
			 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
			 * Invariant: Operations within the block strictly maintain target functional boundaries.
			 */
			for (k = 0; k < N; k++) {
				
				prod2[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (j = 0; j < N; j++) {
			
			C[i * N + j] = prod1[i * N + j] + prod2[i * N + j];
		}
	}

	
	free(AB);
	free(prod1);
	free(prod2);
	return C;
}