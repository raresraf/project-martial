/**
 * @file solver_opt.c
 * @brief A hand-optimized C implementation for solving a matrix equation.
 * @details This file solves the equation C = (A * B) * B' + (A' * A). It is an
 * "optimized" version that attempts to improve performance over a naive implementation
 * by using several C-level optimization techniques. These include using the 'register'
 * keyword as a hint to the compiler, extensive use of pointer arithmetic to reduce
 * indexing overhead, and reordering loops to improve cache locality.
 *
 * @algorithm Loop-based matrix multiplication with manual C-level optimizations.
 * @time_complexity O(N^3), as the fundamental algorithm still relies on triply-nested loops.
 * @space_complexity O(N^2) for storing the intermediate and final result matrices.
 */
#include "utils.h"

/**
 * @brief Solves the matrix equation C = (A * B) * B' + (A' * A) using manually optimized loops.
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the matrix A, stored in row-major order.
 * @param B A pointer to the matrix B, stored in row-major order.
 * @return A pointer to the resulting matrix C. The caller is responsible for freeing this memory.
 */
double* my_solver(int N, double *A, double* B) {
	double *C, *A_tA, *AB;
	// Optimization: Use of 'register' keyword hints the compiler to store these
	// loop counters in CPU registers for potentially faster access.
	register int i, j, k;
	
	C = calloc(N * N, sizeof(double));
	if (C == NULL)
		return NULL;
	A_tA = calloc(N * N, sizeof(double));
	if (A_tA == NULL)
		return NULL;
	AB = calloc(N * N, sizeof(double));
	if (AB == NULL)
		return NULL;
	
	// Block 1: Compute the upper triangle of the symmetric matrix A_tA = A' * A.
	// Optimization: Loops are reordered (k, i, j) and pointers are used extensively
	// to improve data locality and reduce address calculation overhead.
	for (k = 0; k < N; k++) {
		register double *orign_At = &A[k * N];
		for (i = k; i < N; i++) {
			register double *At = orign_At + i;
			register double *pA = orign_At + k;
			register double *pAt_A = &A_tA[i * N + k]; 
			for (j = k; j < N; j++) {
				*pAt_A += *At * *pA;
				pA++;
				pAt_A++;
			}
		}
	}
	
	// Block 2: Compute AB = A * B.
	// Optimization: Assumes A is upper triangular by starting the 'k' loop from 'i'.
	// Pointers are used to iterate through rows and columns.
	for (i = 0; i < N; i++) {
		register double *orign_A = &A[i * N];
		register double *orign_AB = &AB[i * N];
		for (k = i; k < N; k++) {
			register double *pA2 = orign_A + k;
			register double *pB = B + k * N;
			register double *pAB = orign_AB;
			for (j = 0; j < N; j++) {
				*pAB += *pA2 * *pB;
				pB++;
				pAB++;
			}
		}
	}
	
	
	// Block 3: Compute C = AB * B'.
	// Optimization: The 'k' loop is innermost to maximize locality when accessing
	// contiguous elements of AB and B (though B is accessed column-wise here,
	// which is less cache-friendly in a row-major layout).
	for (i = 0; i < N; i++) {
		register double *orign_C = &C[i * N];
		register double *orign_AB = &AB[i * N];
		for (j = 0; j < N; j++) {
			register double *pC = orign_C + j;
			register double *pAB = orign_AB;
			register double *pB = &B[j * N];
			for (k = 0; k < N; k++) {
				// Accesses B' by iterating through rows of B in the outer loops (i, j)
				// and columns of B in the inner loop (k).
				*pC += *pAB * *pB;
				pAB++;
				pB++;
			}
		}
	}
	
	// Block 4: Add the intermediate results: C = C + A_tA.
	// Optimization: A simple loop with pointer arithmetic for element-wise addition.
	for (i = 0; i < N; i++) {
		register double *pC = &C[i * N];
		register double *pA_tA = &A_tA[i * N];
		for (j = 0;  j < N; j++) {
			*pC += *pA_tA;
			pC++;
			pA_tA++;
		}
	}

	free(A_tA);
	free(AB);
	return C;
}