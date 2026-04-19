/**
 * @raw/ad0817ea-cc15-458f-b539-26c872097786/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver computing C = A^T * A + (A * B) * B^T.
 * Algorithm: Standard nested loops generating explicit transposition maps and partial multiplications via symmetric optimizations.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ representing intermediate target buffers.
 */
#include "utils.h"

double* my_solver(int N, double *A, double* B) {

	int i, j, k;
	double *res1, *res2, *res3;
	int min;

	/**
	 * Functional Utility: Initializes dynamically allocated matrix buffers and caches for intermediate outputs.
	 */
	res1 = malloc(N*N*sizeof(double));
	res2 = malloc(N*N*sizeof(double));
	res3 = malloc(N*N*sizeof(double));

	/**
	 * Block Logic: Computes A^T * A internally setting the solution inside res1.
	 * Invariant: Limits iterations relying on upper triangular structure.
	 */
	for (i = 0; i < N; i++) {
   		for (j = 0; j < N; j++) {
      		res1[i * N + j] = 0.0;

      		if (j < i) {
      			min = j;
      		} else {
      			min = i;
      		}

			for (k = 0; k <= min; k++) {
				res1[i * N + j] += A[k * N + i] * A[k * N + j];
      		}
   		}
	}

	/**
	 * Block Logic: Computes the base product res2 = A * B.
	 * Invariant: Operates independently of specific matrix bounds traversing the full NxN block structure.
	 */
	for (i = 0; i < N; i++) {
   		for (j = 0; j < N; j++) {
      		res2[i * N + j] = 0.0;

	      	for (k = 0; k < N; k++) {
				res2[i * N + j] += A[i * N + k] * B[k * N + j];
      		}
   		}
	}

	/**
	 * Block Logic: Evaluates trailing product res2 * B^T defining the vector output inside res3.
	 * Invariant: Extracts parameters directly reflecting matrix B transposition sequentially mapping indices.
	 */
	for (i = 0; i < N; i++) {
   		for (j = 0; j < N; j++) {
      		res3[i * N + j] = 0.0;

	      	for (k = 0; k < N; k++) {
				res3[i * N + j] += res2[i * N + k] * B[j * N + k];
      		}
   		}
	}

	/**
	 * Block Logic: Performs the final reduction translating the formula into res3.
	 */
	for (i = 0; i < N; i++) {
   		for (j = 0; j < N; j++) {
      		res3[i * N + j] += res1[i * N + j];

   		}
	}

	free(res1);
	free(res2);
	

	return res3;

	printf("NEOPT SOLVER\n");
	return NULL;
}
