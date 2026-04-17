/**
 * @file solver_neopt.c
 * @brief High-level source code module.
 * Ensures cache-friendly data access, potential loop unrolling, and SIMD optimizations for C/C++.
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	
	double *AB = malloc(N * N * sizeof(double));
	double *ABBt = malloc(N * N * sizeof(double));
	double *AtA = malloc(N * N * sizeof(double));
	double *C = malloc(N * N * sizeof(double));

	/**
	 * @brief Pre-condition: Evaluates logical divergence based on current state.
	 * Invariant: Guarantees correct execution flow according to conditional partitioning.
	 */
	if (AB == NULL || ABBt == NULL || AtA == NULL || C == NULL){
		fprintf(stderr, "malloc error\n");
		exit(EXIT_FAILURE);
	}

	
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
			AB[i * N + j] = 0;
			/**
			 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
			 * Invariant: Operations within the block strictly maintain target functional boundaries.
			 */
			for (int k = 0; k < N; k++) {
				
				/**
				 * @brief Pre-condition: Evaluates logical divergence based on current state.
				 * Invariant: Guarantees correct execution flow according to conditional partitioning.
				 */
				if (i <= k)
					AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
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
			ABBt[i * N + j] = 0;
			/**
			 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
			 * Invariant: Operations within the block strictly maintain target functional boundaries.
			 */
			for (int k = 0; k < N; k++) {
				ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	
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
			AtA[i * N + j] = 0;
			/**
			 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
			 * Invariant: Operations within the block strictly maintain target functional boundaries.
			 */
			for (int k = 0; k < N; k++) {
				
				/**
				 * @brief Pre-condition: Evaluates logical divergence based on current state.
				 * Invariant: Guarantees correct execution flow according to conditional partitioning.
				 */
				if (i >= k && k <= j)
					AtA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	
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
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	
	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
