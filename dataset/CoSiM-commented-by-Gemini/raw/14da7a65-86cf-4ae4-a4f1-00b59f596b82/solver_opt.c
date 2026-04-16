/**
 * @file solver_opt.c
 * @brief A manually micro-optimized implementation of a matrix equation solver.
 * @details This file computes the solution to the matrix equation:
 * result = (A * B) * B^T + A^T * A. This version attempts to optimize the
 * computation by explicitly transposing the input matrices first, and then
 * using pointer arithmetic extensively to perform the multiplications. The
 * core algorithm remains O(N^3).
 */
#include "utils.h"
#include <stdlib.h>

/**
 * @brief Computes (A * B) * B^T + A^T * A with manual pointer-based optimizations and explicit transposition.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N, row-major, upper triangular).
 * @param B A pointer to the second input matrix (N x N, row-major).
 * @return A pointer to the resulting N x N matrix (named C). The caller is responsible for freeing this memory.
 */
double* my_solver(int N, double *A, double* B) {
	double *C;
	double *A_t;
	double *B_t;
	double *AB;
	register int i, j, k;
	register double *A_t_ptr, *B_t_ptr;
	register double *A_ptr, *B_ptr;
	register double *A_tAptr;
	register double *pa, *pb, result;
	register double *AB_ptr;
	register double *C_ptr;

	// Allocate memory for transposed matrices and intermediate results.
	A_t = (double *)calloc(N * N, sizeof(double));
	if (NULL == A_t)
		exit(EXIT_FAILURE);
	
	B_t = (double *)calloc(N * N, sizeof(double));
	if (NULL == B_t)
		exit(EXIT_FAILURE);
	
	AB = (double *)calloc(N * N, sizeof(double));
	if (NULL == AB)
		exit(EXIT_FAILURE);
	
	C = (double *)calloc(N * N, sizeof(double));
	if (NULL == C)
		exit(EXIT_FAILURE);

	/**
	 * Block Logic: Explicitly transpose matrices A and B.
	 * This costs O(N^2) time and additional memory but can simplify
	 * the memory access patterns in the multiplication loops.
	 */
	for (i = 0; i != N; ++i) {
		A_t_ptr = A_t + i;  
		B_t_ptr = B_t + i;  

		A_ptr = A + i * N;  
		B_ptr = B + i * N;  

		for (j = 0; j != N; ++j) {
			*A_t_ptr = *A_ptr;
			*B_t_ptr = *B_ptr;
			A_t_ptr += N;
			B_t_ptr += N;
			++A_ptr;
			++B_ptr;
		}
	}

	
	/**
	 * Block Logic: Compute C = A^T * A.
	 * This multiplies the rows of A_t (columns of A) together.
	 * Assumes A is upper triangular (k <= i).
	 * Time Complexity: O(N^3)
	 */
	for (i = 0; i != N; ++i) {
		A_tAptr = C + i * N;
		A_t_ptr = A_t + i * N;

		for (j = 0; j != N; ++j) {
			pa = A_t_ptr;
			pb = A_t + j * N;
			result = 0;

			for (k = 0; k <= i; ++k) {
				result += *pa * *pb;
				++pa;
				++pb;
			}

			*A_tAptr = result;
			++A_tAptr; 
		}
	}

	
	/**
	 * Block Logic: Compute AB = A * B.
	 * This multiplies rows of A with columns of B (which are rows of B_t).
	 * Assumes A is upper triangular (k starts from i).
	 * Time Complexity: O(N^3)
	 */
	for (i = 0; i != N; ++i) {
		AB_ptr = AB + i * N;
		A_ptr = A + i * N;

		for (j = 0; j != N; ++j) {
			pa = A_ptr + i;
			pb = B_t + j * N + i;
			result = 0;

			for (k = i; k < N; ++k) {
				result += *pa * *pb;
				++pa;
				++pb;
			}

			*AB_ptr = result;
			++AB_ptr; 
		}
	}

	
	/**
	 * Block Logic: Compute C += (A * B) * B^T
	 * This calculates the dot product of rows from AB and rows from B,
	 * which is equivalent to AB * B^T, and adds it to the existing values in C.
	 * Time Complexity: O(N^3)
	 */
	for (i = 0; i != N; ++i) {
		C_ptr = C + i * N;
		AB_ptr = AB + i * N;

		for (j = 0; j != N; ++j) {
			pa = AB_ptr;
			pb = B + j * N;
			result = 0;

			for (k = 0; k != N; ++k) {
				result += *pa * *pb;
				++pa;
				++pb;
			}

			*C_ptr += result;
			++C_ptr; 
		}
	}

	// Free all allocated intermediate and transposed matrices.
	free(A_t);
	free(B_t);
	free(AB);
	return C;	
}