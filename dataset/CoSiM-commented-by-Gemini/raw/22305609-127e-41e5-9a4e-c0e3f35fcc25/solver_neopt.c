/**
 * @file solver_neopt.c
 * @brief A non-optimized, baseline implementation of a matrix solver.
 * @details This file provides a straightforward, unoptimized solution for the matrix equation
 * C = (A * B) * B' + A' * A, where A is an upper triangular matrix. The implementation
 * uses explicit loops and pointer arithmetic for all operations.
 */
#include "utils.h"
#include <string.h>


/**
 * @brief Solves C = (A * B) * B' + A' * A using a non-optimized, sequential approach.
 * @param N The dimension of the square matrices.
 * @param A A pointer to the input upper triangular matrix A (N x N).
 * @param B A pointer to the input matrix B (N x N).
 * @return A pointer to the resulting matrix C (N x N).
 *
 * @details The function performs the matrix operations in several distinct steps, using
 * temporary buffers to store intermediate results.
 * 1. Manually computes the transpose of A and B into `transA` and `transB`.
 * 2. Computes `C = A * B`, exploiting the upper triangular nature of A.
 * 3. Copies the result to a temporary buffer `tmp`.
 * 4. Computes `C = tmp * transB`, which is `(A * B) * B'`.
 * 5. Copies the result to `tmp` again.
 * 6. Computes `tmp2 = transA * A`, which is `A' * A`.
 * 7. Computes the final result `C = tmp + tmp2`.
 */
double *my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER
");
	size_t i, j, k;

	// Allocate memory for the final result and intermediate matrices.
	double *C = calloc(sizeof(double), N * N);
	double *transA = calloc(sizeof(double), N * N);
	double *transB = calloc(sizeof(double), N * N);

	/**
	 * Block Logic: Step 1: Manually compute transposes of A and B.
	 * Time Complexity: O(N^2)
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*(transA + N * i + j) = *(A + N * j + i);
			*(transB + N * i + j) = *(B + N * j + i);
		}
	}

	
	/**
	 * Block Logic: Step 2: Compute the intermediate product C = A * B.
	 * The inner loop `k` starts from `i`, which is an optimization for
	 * multiplication with an upper triangular matrix A.
	 * Time Complexity: O(N^3), with a lower constant factor.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*(C + N * i + j) = 0.0;
			for (k = i; k < N; ++k) {
				*(C + N * i + j) += *(A + N * i + k) * *(B + N * k + j);
			}
		}
	}

	double *tmp = calloc(sizeof(double), N * N);
	memcpy(tmp, C, N * N * sizeof(double));

	
	/**
	 * Block Logic: Step 3: Compute the first term C = (A * B) * B'.
	 * This multiplies the intermediate result `tmp` (which is A*B) with `transB`.
	 * Time Complexity: O(N^3)
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*(C + N * i + j) = 0.0;
			for (k = 0; k < N; ++k) {
				*(C + N * i + j) += *(tmp + N * i + k) * *(transB + N * k + j);
			}
		}
	}

	memcpy(tmp, C, N * N * sizeof(double));
	

	double *tmp2 = calloc(sizeof(double), N * N);
	
	/**
	 * Block Logic: Step 4: Compute the second term tmp2 = A' * A.
	 * This multiplies `transA` with `A`.
	 * Time Complexity: O(N^3)
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*(tmp2 + N * i + j) = 0.0;
			for (k = 0; k < N; ++k) {
				*(tmp2 + N * i + j) += *(transA + N * i + k) * *(A + N * k + j);
			}
		}
	}
	
	/**
	 * Block Logic: Step 5: Compute the final result C = ((A * B) * B') + (A' * A).
	 * This is an element-wise addition of the two terms stored in `tmp` and `tmp2`.
	 * Time Complexity: O(N^2)
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*(C + N * i + j) = *(tmp + N * i + j) + *(tmp2 + N * i + j);
		}
	}

	free(tmp);
	free(tmp2);
	free(transA);
	free(transB);

	return C;
}
