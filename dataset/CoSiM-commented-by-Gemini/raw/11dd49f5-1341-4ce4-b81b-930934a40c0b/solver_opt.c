/**
 * @file solver_opt.c
 * @brief A manually micro-optimized implementation of a matrix equation solver.
 * @details This file computes the solution to the matrix equation:
 * result = (A * B) * B^T + A^T * A. It is a variant of the naive, loop-based
 * solver but attempts to optimize performance by using pointer arithmetic
 * instead of array indexing and liberal use of the 'register' keyword.
 * These are micro-optimizations that may offer slight performance gains on
 * older compilers but are generally less effective than algorithmic changes
 * or using a dedicated library like BLAS. The core algorithm remains O(N^3).
 */
#include "utils.h"
#include <string.h>

/**
 * @brief Computes (A * B) * B^T + A^T * A with manual pointer-based optimizations.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N, row-major, upper triangular).
 * @param B A pointer to the second input matrix (N x N, row-major).
 * @return A pointer to the resulting N x N matrix. The caller is responsible for freeing this memory.
 */
double* my_solver(int N, double *A, double* B) {
	// Allocate memory for intermediate and final result matrices.
	// The 'register' keyword is a hint for the compiler to keep variables in CPU registers.
	register double *ab = (double *) malloc(N * N * sizeof(double));
	register double *abbt = (double *) malloc(N * N * sizeof(double));
	register double *ata = (double *) malloc(N * N * sizeof(double));
	register double *result =  (double *) malloc(N * N * sizeof(double));

	
	/**
	 * Block Logic: Compute ab = A * B using pointer arithmetic.
	 * Assumes A is an upper triangular matrix.
	 * Optimization: Pointers are used to iterate through rows and columns,
	 * potentially reducing index calculation overhead.
	 * Time Complexity: O(N^3)
	 */
	for (int i = 0; i < N; ++i) {
		register double *a_i = A + i * N;
		for (int j = 0; j < N; ++j) {			
			register double *aux_a_i = a_i;
			register double *b_j = B + j;
			register double sum = 0.0;

			b_j += i * N;

			for (int k = i; k < N; ++k) {
				sum += *(aux_a_i + i) * *b_j;
				aux_a_i++;
				b_j += N;
			}
			ab[i * N + j] = sum;
		}
	}

	
	/**
	 * Block Logic: Compute abbt = ab * B^T using pointer arithmetic.
	 * Accessing B^T is done by iterating through B in a column-major fashion
	 * within the inner loop.
	 * Time Complexity: O(N^3)
	 */
	for (int i = 0; i < N; ++i) {
		register double *ab_in = ab + i * N;
		for (int j = 0; j < N; ++j) {
			register double *aux_ab_in = ab_in;
			register double *b_jn = B + j * N;
			register double sum = 0.0;

			for (int k = 0; k < N; ++k) {
				sum += *aux_ab_in * *b_jn;
				++aux_ab_in;
				++b_jn;
			}
			abbt[i * N + j] = sum;
		}
	}

	
	/**
	 * Block Logic: Compute ata = A^T * A using pointer arithmetic.
	 * This performs a dot product between columns of A.
	 * Assumes A is upper triangular.
	 * Time Complexity: O(N^3)
	 */
	for (int i = 0; i < N; ++i) {
		register double *a_i = A + i;
		for (int j = 0; j < N; ++j) {
			register double *aux_a_i = a_i;
			register double *aux_a_j = A + j;
			register double sum = 0.0;
			
			for (int k = 0; k <= i; ++k) {
				sum += *aux_a_i * *aux_a_j;
				aux_a_i += N;
				aux_a_j += N;
			}
			ata[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Final summation.
	 * result = abbt + ata => (A * B) * B^T + A^T * A
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			result[i * N + j] = abbt[i * N + j] + ata[i * N + j];
		}
	}
	
	// Free the memory allocated for intermediate matrices.
	free(ab);
	free(abbt);
	free(ata);
	return result;	
}
