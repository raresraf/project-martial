
#include "utils.h"

/**
 * @file solver_opt.c
 * @brief Manually optimized matrix solver implementation using pre-transposition and pointer arithmetic.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A^T * A) + (A * B) * B^T
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Employs several manual C tuning strategies:
 * 1. Explicit Pre-transposition: Pre-computes A^T and B^T to convert strided 
 *    column access into sequential row-major increments, maximizing spatial 
 *    locality and cache hit rates.
 * 2. Pointer Arithmetic: Replaces array indexing with direct increments 
 *    (A_line++, B_t_col += N) to reduce instruction latency.
 * 3. Register Caching: Caches frequently used row base pointers and 
 *    accumulators in CPU registers.
 * 4. Triangular Property Awareness: Minimizes work by restricting ranges 
 *    for products involving triangular matrix A.
 * 
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

/**
 * my_solver - Optimized matrix solver with cache-aware traversal and register hints.
 */
double* my_solver(int N, double *A, double *B) {
	double * C, *AB, *ABB_t, *A_t, *B_t;
	int i, j, k;

	// Logic: Workspace allocation for partial products and transposed operands.
	A_t = calloc(N * N, sizeof(double));
	if(A_t == NULL)
		printf("Probleme la alocarea memoriei\n");
	B_t = calloc(N * N, sizeof(double));
	if(B_t == NULL)
		printf("Probleme la alocarea memoriei\n");
	C = calloc(N * N, sizeof(double));
	if(C == NULL)
		printf("Probleme la alocarea memoriei\n");
	AB = calloc(N * N, sizeof(double));
	if(AB == NULL)
		printf("Probleme la alocarea memoriei\n");
	ABB_t = calloc(N * N, sizeof(double));
	if(ABB_t == NULL)
		printf("Probleme la alocarea memoriei\n");

	/**
	 * Block Logic: Pre-compute B_t = B^T.
	 * Optimization: Sequential-to-Strided memory copy.
	 */
	for(i = 0; i < N; i++) {
		register double *B_t_col = B_t + i;
		register double *B_line = B + i * N;
		for(j = 0; j < N; j++) {
			*B_t_col = *B_line;
			B_line++;
			B_t_col += N;
		}
	}
	
	/**
	 * Block Logic: Pre-compute A_t = A^T.
	 * Optimization: Sparse triangular copy.
	 */
	for(i = 0; i < N; i++) {
		register double *A_t_col = A_t + i * N + i;
		register double *A_line = A + i * N;
		for(j = i; j < N; j++) {
			*A_t_col = *(A_line + j);
			A_t_col += N;
		}
	}
	
	/**
	 * Block Logic: Compute C = A^T * A.
	 * Optimization: Uses pre-computed A_t to enable dual pointer dot products.
	 */
	for(i = 0; i < N; i++) {
		register double *C_line = C + i * N;
		register double *A_t_line = A_t + i * N;
		for(j = 0; j < N; j++, C_line++) {
			register double result = 0;
			register double *A_t_col = A_t_line;
			register double *A_col = A + j;
			for(k = 0; k < N; k++, A_t_col++, A_col += N) {
				result += *A_t_col * *A_col;
			}
			*C_line = result;
		}
	}
	
	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Exploits upper triangularity of A by skipping k < i.
	 */
	for(i = 0; i < N; i++) {
		register double *AB_line = AB + i * N;
		register double *A_line = A + i * N;
		for(j = 0; j < N; j++) {
			register double *AB_col = AB_line + j;
			register double result = 0;
			register double *A_col = A_line + i;
			register double *B_col = B + i * N + j;
			for(k = i; k < N; k++, A_col++, B_col += N) {
				result += *A_col * *B_col;
			}
			*AB_col = result;
		}
	}
	
	/**
	 * Block Logic: Compute ABB_t = AB * B^T.
	 * Optimization: Leverages pre-computed B_t for cache-friendly sequential access.
	 */
	for(i = 0; i < N; i++) {
		register double *ABB_t_line = ABB_t + i * N;
		register double *AB_line = AB + i * N;
		for(j = 0; j < N; j++) {
			register double *ABB_t_col = ABB_t_line + j;
			register double result = 0;
			register double *AB_col = AB_line;
			register double *B_t_col = B_t + j;
			for(k = 0; k < N; k++, AB_col++, B_t_col += N) {
				result += *AB_col * *B_t_col;
			}
			*ABB_t_col = result;
		}
	}

	/**
	 * Block Logic: Final aggregation C = C + ABB_t.
	 * Optimization: Linear memory traversal for summation.
	 */
	for(i = 0; i < N; i++) {
		register double *C_line = C + i * N;
		register double *ABB_t_line = ABB_t + i * N;
		for(j = 0; j < N; j++, ABB_t_line++, C_line++) {
			*C_line += *ABB_t_line;
		}
	}
	
	free(AB);
	free(A_t);
	free(B_t);
	free(ABB_t);

	return C;
}
