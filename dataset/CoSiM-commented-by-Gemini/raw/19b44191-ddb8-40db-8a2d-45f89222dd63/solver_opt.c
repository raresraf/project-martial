/**
 * @file solver_opt.c
 * @brief A manually micro-optimized implementation of a matrix equation solver.
 * @details This file computes the solution to the matrix equation:
 * C = A^T * A + A * (B * B^T). This version attempts to optimize performance
 * by explicitly transposing matrix A and then using pointer arithmetic and the
 * 'register' keyword to perform the multiplications. The core algorithm remains O(N^3).
 */
#include "utils.h"
#include <stdlib.h>

/**
 * @brief Computes C = A^T * A + A * (B * B^T) with manual optimizations.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N, row-major).
 * @param B A pointer to the second input matrix (N x N, row-major).
 * @return A pointer to the resulting N x N matrix (C). The caller is responsible for freeing this memory.
 */
double* my_solver(int N, double *A, double* B) {
	register double *rez, *C, *A_trans;
	register int i, j, k;
	
	// Allocate and zero-initialize memory for results and intermediate matrices.
	C = calloc(N * N, sizeof(double));
	rez = calloc(N * N, sizeof(double));
	A_trans = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Explicitly transpose matrix A into A_trans.
	 * This costs O(N^2) time and extra memory but may simplify access patterns
	 * for the A^T * A calculation later.
	 */
	for (i = 0; i < N; ++i) {
		register double *pointer_trans = A_trans + i;
		register double *pointer = A + i * N;
		for (j = 0; j < N; ++j) {
			*pointer_trans = *pointer;
			pointer_trans += N;
			pointer++;
		}
	}
	
	/**
	 * Block Logic: Compute rez = B * B^T.
	 * This is done by calculating the dot product of each row of B with every
	 * other row of B.
	 * Time Complexity: O(N^3)
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pa = B + i * N;
		for (j = 0; j < N; ++j) {
			register double suma = 0.0;
			register double *pa = orig_pa; 
			register double *pb = B + j * N;
			for (k = 0; k < N; ++k) {
				suma += *pa * *pb;
				pa++;
				pb++;
			}
			rez[i * N + j] = suma;
		}
	}

	
	/**
	 * Block Logic: Compute C = A * rez, which is A * (B * B^T).
	 * This performs a standard matrix multiplication using pointers.
	 * Time Complexity: O(N^3)
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pa = A + i * N;
		for (j = 0; j < N; ++j) {
			register double suma = 0.0;
			register double *pa = orig_pa; 
			register double *pb = rez + j;
			for (k = 0; k < N; ++k) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			C[i * N + j] = suma;
		}
	}

	
	/**
	 * Block Logic: Compute C += A^T * A.
	 * This performs matrix multiplication of the transposed A (A_trans) with the
	 * original A and adds the result to C.
	 * Time Complexity: O(N^3)
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pa = A_trans + i * N;
		for (j = 0; j < N; ++j) {
			register double suma = 0.0;
			register double *pa = orig_pa; 
			register double *pb = A + j;
			for (k = 0; k < N; ++k) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			C[i * N + j] += suma;
		}
	}

	// Free the allocated intermediate matrices.
	free(A_trans);
	free(rez);
	return C;
}
