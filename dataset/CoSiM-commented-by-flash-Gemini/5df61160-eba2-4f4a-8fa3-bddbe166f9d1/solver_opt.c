/**
 * @5df61160-eba2-4f4a-8fa3-bddbe166f9d1/solver_opt.c
 * @brief Optimized manual implementation of a matrix expression solver.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using pointer arithmetic, 
 * register variables, and local sum accumulation to enhance cache locality and reduce 
 * memory latency.
 * Domain: HPC Performance Tuning.
 */

#include "utils.h"


/**
 * @brief Manually optimized matrix solver kernel.
 * Optimization: Replaces array indexing with pointer arithmetic to minimize address 
 * calculation overhead in the inner loops.
 */
double* my_solver(int N, double *A, double* B) {
	double *RESULT = (double *) calloc(N * N, sizeof(double));
	double *TEMPORARY = (double *) calloc(N * N, sizeof(double));
	double *initial_line_parser, *line_parser, *column_parser;
	register double temporary_sum;
	register int i, j, k;

	/**
	 * Block Logic: Compute TEMPORARY = A * B.
	 * Optimization: Uses `line_parser` and `column_parser` pointers to sweep through 
	 * matrices linearly, improving prefetcher efficiency.
	 */
	for(i = 0; i < N; ++i){
		initial_line_parser = &A[i * N];
		for(j = 0; j < N; ++j){
			line_parser = initial_line_parser;
			column_parser = &B[j];
			temporary_sum = 0;
			for(k = 0; k < N; ++k, ++line_parser, column_parser += N){
				temporary_sum += *line_parser * *column_parser;
			}
			TEMPORARY[i * N + j] = temporary_sum;
		}
	}

	/**
	 * Block Logic: Compute RESULT += TEMPORARY * B^T.
	 * Optimization: Scalar promotion of the sum accumulator into a register.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			temporary_sum = 0;
			for (k = 0; k < N; ++k) {
				temporary_sum += TEMPORARY[i * N + k] * B[j * N + k];
			}
			RESULT[i * N + j] += temporary_sum;
		}
	}

	/**
	 * Block Logic: Compute RESULT += A^T * A.
	 * Optimization: Computes the column-wise dot product of matrix A.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			temporary_sum = 0;
			for (k = 0; k < N; ++k) {
				temporary_sum += A[k * N + i] * A[k * N + j];
			}
			RESULT[i * N + j] += temporary_sum;
		}
	}

	free(TEMPORARY);
	return RESULT;
}
