/**
 * @file solver_opt.c
 * @brief High-level source code module.
 * Ensures cache-friendly data access, potential loop unrolling, and SIMD optimizations for C/C++.
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	register int i, j, k;

	
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
		
		
		register double* lineA = A + i * N + i;
		register double* pAB = AB + i * N;
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			register double* elemA = lineA;
			register double* columnB = B + i * N + j;
			/**
			 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
			 * Invariant: Operations within the block strictly maintain target functional boundaries.
			 */
			for (k = i; k < N; k++) {
				
				sum += *elemA * *columnB;
				elemA++;
				columnB += N;
			}
			*pAB = sum;
			pAB++;
		}
	}

	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (i = 0; i < N; i++) {
		register double* lineAB = AB + i * N;
		register double* pProd1 = prod1 + i * N;
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			register double* elemAB = lineAB;
			register double* elemBt = B + j * N;
			/**
			 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
			 * Invariant: Operations within the block strictly maintain target functional boundaries.
			 */
			for (k = 0; k < N; k++) {
				
				sum += *elemAB * *elemBt;
				elemAB++;
				elemBt++;
			}
			*pProd1 = sum;
			pProd1++;
		}
	}

	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (i = 0; i < N; i++) {
		register double* columnAt = A + i;
		register double* pProd2 = prod2 + i * N;
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			register double* elemAt = columnAt;
			register double* elemA = A + j;
			/**
			 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
			 * Invariant: Operations within the block strictly maintain target functional boundaries.
			 */
			for (k = 0; k < N; k++) {
				
				sum += *elemAt * *elemA;
				elemA += N;
				elemAt += N;
			}
			*pProd2 = sum;
			pProd2++;
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