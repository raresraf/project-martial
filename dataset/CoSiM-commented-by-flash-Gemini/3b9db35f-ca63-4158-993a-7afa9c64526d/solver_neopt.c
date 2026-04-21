
#include "utils.h"

/**
 * @file solver_neopt.c
 * @brief Unoptimized matrix solver with manual triangularity dispatch.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Algorithm: Standard triple-nested loop matrix multiplication.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store multiple intermediate matrices.
 * 
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

/**
 * multiply - Generic matrix multiplication with specialized triangular handling.
 * 
 * @param tr Flag for optimization: 1 for upper triangular A, 2 for lower triangular B, 0 for general.
 * 
 * Logic: Reduces the number of operations by skipping dot products known to 
 * involve zeroes in triangular matrices.
 */
double* multiply(int N, double *A, double *B, int tr){
	int i = 0, j = 0, k = 0;
	double *mul;
	mul = (double *)calloc(N * N, sizeof(double));
	if (!mul) return NULL;

	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			for (k = 0; k < N; k++){
				// Block Logic: Optimization Dispatch.
				if (tr == 1){
					// Case: A is upper triangular; skip k < i.
					if (i <= k)
						mul[i * N + j] += A[i * N + k] * B[k * N + j];
					else continue;
				}
				else if (tr == 2){
					// Case: B is lower triangular; skip k > j.
					if (k <= j)
						mul[i * N + j] += A[i * N + k] * B[k * N + j];
					else continue;
				}
				else mul[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
	return mul;
}

/**
 * my_solver - Naive implementation using manual transpositions and modular multiplication.
 */
double* my_solver(int N, double *A, double* B) {
	double *At, *Bt, *sum, *sum2, *sum3, *c;
	int i = 0, j = 0;
	At = (double *)calloc(N * N, sizeof(double));
	Bt = (double *)calloc(N * N, sizeof(double));
	c = (double *)calloc(N * N, sizeof(double));
	
	/**
	 * Block Logic: Compute At = A^T.
	 * Optimization: Exploits upper triangularity; only copies lower half.
	 */
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			if (i >= j){
				At[i * N + j] = A[j * N + i];
			}
		}
	}
	
	/**
	 * Block Logic: Compute Bt = B^T.
	 */
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			Bt[i * N + j] = B[j * N + i];
		}
	}
	
	// Step 1: sum = A * B (A is upper triangular).
	sum = multiply(N, A, B, 1);
	
	// Step 2: sum2 = sum * Bt (General multiplication).
	sum2 = multiply(N, sum, Bt, 0);
	
	// Step 3: sum3 = At * A (At is lower triangular).
	sum3 = multiply(N, At, A, 2);
	
	/**
	 * Block Logic: Final result aggregation.
	 */
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			c[i * N + j] = sum2[i * N + j] + sum3[i * N + j];	
		}
	}

	free(At);
	free(Bt);
	free(sum);
	free(sum2);
	free(sum3);
	return c;
}
