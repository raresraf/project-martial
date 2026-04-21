/**
 * @file solver_opt.c
 * @brief Manually optimized Implementation of the matrix expression solver.
 *
 * Computes: Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 *
 * Performance Optimization:
 * 1. Loop Unrolling: Processes multiple elements per iteration to reduce 
 *    branching overhead and exploit Instruction-Level Parallelism (ILP).
 * 2. Pointer Arithmetic: Uses raw address increments instead of array 
 *    indexing to avoid redundant base-address calculations.
 * 3. Register Caching: Caches frequently used sums and row base pointers 
 *    in CPU registers.
 * 4. Cache-Aware Access: Iterates through memory in patterns that maximize 
 *    spatial locality.
 *
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

#include "utils.h"

/**
 * my_solver - Optimized implementation using loop unrolling and pointer arithmetic.
 */
double* my_solver(int N, double *A, double *B) {
	printf("OPT SOLVER\n");
	
	/**
	 * Pre-condition: Allocation of result and scratchpad buffers.
	 */
	double *AB = calloc(N * N, sizeof(double));
	double *ABBt = malloc(N * N, sizeof(double));
	double *AtA = calloc(N * N, sizeof(double));
	double *C = malloc(N * N * sizeof(double));

	if (AB == NULL || ABBt == NULL || AtA == NULL || C == NULL){
		fprintf(stderr, "malloc error\n");
		exit(EXIT_FAILURE);
	}

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Exploits triangularity of A and applies loop unrolling (factor of 4) 
	 * on the innermost dimension to improve throughput.
	 */
	for (int i = 0; i < N; i++) {
		double *copy_res = &(AB[i * N]);
		for (int k = 0; k < N; k++) {
			double *line = &(A[i * N + k]);
			double *col = &(B[k * N]);
			double *res = copy_res;
			
			if (i <= k) {
				for (int j = 0; j < N; j += 4) {
					// Inline: Hand-unrolled vector multiply-add.
					*res += *line * *col; col++; res++;
					*res += *line * *col; col++; res++;
					*res += *line * *col; col++; res++;
					*res += *line * *col; col++; res++;
				}
			}
		}
	}

	/**
	 * Block Logic: Compute ABBt = AB * B^T.
	 * Optimization: Uses a register accumulator and unrolled dot product logic.
	 */
	for (int i = 0; i < N; i++) {
		double *copy_line = &(AB[i * N]);
		for (int j = 0; j < N; j++) {
			double *line = copy_line;
			double *col = &(B[j * N]);
			register double sum = 0.0;
			for (int k = 0; k < N; k += 4) {
				sum += *line * *col; line++; col++;
				sum += *line * *col; line++; col++;
				sum += *line * *col; line++; col++;
				sum += *line * *col; line++; col++;
			}
			ABBt[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Compute AtA = A^T * A.
	 * Optimization: Exploits sparse upper triangular structure to skip redundant checks.
	 */
	for (int k = 0; k < N; k++) {
		double *copy_col = &(A[k * N]);
		for (int i = 0; i < N; i++) {
			double *res = &(AtA[i * N]);
			double *line = &(A[k * N + i]);
			double *col = copy_col;

			if (i < k) {
				continue;
			}
			for (int j = 0; j < N; j++) {
				if (k <= j) {
					*res += *line * *col;
				}
				col++;
				res++;
			}
		}
	}

	/**
	 * Block Logic: Final unrolled aggregation Result = ABBt + AtA.
	 */
	double *a = &(ABBt[0]);
	double *b = &(AtA[0]);
	double *c = &(C[0]);
	for (int i = 0; i < N * N; i += 4) {
		*c = *a + *b; a++; b++; c++;
		*c = *a + *b; a++; b++; c++;
		*c = *a + *b; a++; b++; c++;
		*c = *a + *b; a++; b++; c++;
	}

	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
