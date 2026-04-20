/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
 */

#include "utils.h"

#define LOOP_UNROLLING 4


double *sum(int N, double *X, double *Y) {
	int i = 0;
	double *sum_result = malloc(N * N * sizeof(double));

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(i = 0; i < N * N; i ++) {
		sum_result[i] = X[i] + Y[i];
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
		double *ptr_res = mul_result + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		double *ptr_b = B + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for(j = 0; j < N; j++) {
			double *ptr_bt = B + j * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for(k = 0; k < N; k += LOOP_UNROLLING) {
				
				
				sum += *(ptr_b + k) * *(ptr_bt + k);
				sum += *(ptr_b + k + 1) * *(ptr_bt + k + 1);
				sum += *(ptr_b + k + 2) * *(ptr_bt + k + 2);
				sum += *(ptr_b + k + 3) * *(ptr_bt + k + 3);
			}
			*(ptr_res + j) += sum;
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
		double *ptr_res = mul_result + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		double *ptr_a = A + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for(k = i; k < N; k++) {
			double *ptr_mat = MAT + k * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register const double val_a = *(ptr_a + k);
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for(j = 0; j < N; j += LOOP_UNROLLING) {
				
				
				*(ptr_res + j) += val_a * *(ptr_mat + j);
				*(ptr_res + j + 1) += val_a * *(ptr_mat + j + 1);
				*(ptr_res + j + 2) += val_a * *(ptr_mat + j + 2);
				*(ptr_res + j + 3) += val_a * *(ptr_mat + j + 3);
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
		double *ptr_res = mul_result + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		double *ptr_at = A + i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for(k = 0; k <= i; k++) {
			double *ptr_a = A + k * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register const double val_at = *(ptr_at + k * N);
			j = k;
			/**
			 * Block Logic: State-driven evaluation.
			 * Invariant: The loop condition reliably gates the execution state.
			 */
			while(j % LOOP_UNROLLING != 0) {
				*(ptr_res + j) += val_at * *(ptr_a + j);
				j++;
			}
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for(; j < N; j += LOOP_UNROLLING) {
				
				
				*(ptr_res + j) += val_at * *(ptr_a + j);
				*(ptr_res + j + 1) += val_at * *(ptr_a + j + 1);
				*(ptr_res + j + 2) += val_at * *(ptr_a + j + 2);
				*(ptr_res + j + 3) += val_at * *(ptr_a + j + 3);
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
