/**
 * @raw/8fd042d2-01f7-415a-9a1e-2f2dba3f9f73/solver_opt.c
 * @brief Optimized dense matrix multiplication solver computing C = A^T * A + (A * B) * B^T.
 * Algorithm: Improves memory access latency by precomputing transposed matrices and utilizing pointer dereferencing for multiplication.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ due to auxiliary storage required for transposed layout buffers.
 */
#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	register int i, j, k;
	
	/**
	 * Functional Utility: Provisions intermediary matrices allocated to 0 to prevent accumulation bugs.
	 */
	double *temp = (double *) calloc(N * N, sizeof(double));
	double *temp1 = (double *) calloc(N * N, sizeof(double));
	double *temp2 = (double *) calloc(N * N, sizeof(double));
	double *res = (double *) calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Reorders matrix memory footprints by populating transposed duplicates of A and B.
	 * Invariant: Generates `temp = A^T` and `temp1 = B^T` concurrently to optimize later sequential row reads.
	 */
	for (i = 0; i < N; ++i) {
		register double *A_t_ptr = temp + i;
                register double *B_t_ptr = temp1 + i;
                register double *A_ptr = A + i * N;
		register double *B_ptr = B + i * N;
		for (j = 0; j < N; ++j, A_t_ptr += N, B_t_ptr += N, ++A_ptr, ++B_ptr) {
			*A_t_ptr = *A_ptr;
			*B_t_ptr = *B_ptr;
		}
	}

	/**
	 * Block Logic: Evaluates A^T * A by executing dot products of row representations.
	 * Optimization: Exploits sequential pointer access to read elements. Limits upper iterations to `i`.
	 */
	for (i = 0; i < N; ++i) {
		register double *temp_copy = temp + i * N;

		for (j = 0; j < N; ++j) {
			register double result = 0;
			register double *A_t_ptr = temp_copy;
			register double *A_ptr = A + j;

			for (k = 0; k <= i; ++k, ++A_t_ptr, A_ptr += N) 
				result += *A_t_ptr * *A_ptr;

			res[i * N + j] = result;
		}
	}
	
	/**
	 * Block Logic: Performs A * B using pointers while saving the outcome into `temp2`.
	 */
	for (i = 0; i < N; ++i) {
		register double *A_copy = A + i * N;
		for (j = 0; j < N; ++j) {
			register double result = 0;
			register double *A_ptr = A_copy;
			register double *B_ptr = B + j;

			for (k = 0; k < N; ++k, ++A_ptr, B_ptr += N) {
				result += *A_ptr * *B_ptr;
			}	

			temp2[i * N + j] = result;
		}
	}
	
	/**
	 * Block Logic: Computes temp2 * B^T and aggregates the answer inside `res` which initially held A^T * A.
	 * Optimization: Takes advantage of the transposed cache structure `temp1` built previously.
	 */
	for (i = 0; i < N; ++i) {
		register double *temp2_copy = temp2 + i * N;

		for (j = 0; j < N; ++j) {
			register double result = 0;
			register double *temp2_ptr = temp2_copy;
			register double *B_t_ptr = temp1 + j;

			for (k = 0; k < N; ++k, ++temp2_ptr, B_t_ptr += N) {
				result += *temp2_ptr * *B_t_ptr;
			}

			res[i * N + j] += result;
		}
	}

	free(temp);
	free(temp1);
	free(temp2);
	
	return res;	
}
