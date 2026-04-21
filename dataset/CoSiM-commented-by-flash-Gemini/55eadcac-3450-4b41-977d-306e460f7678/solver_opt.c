/**
 * @file solver_opt.c
 * @brief Manually optimized Implementation of the matrix expression solver.
 *
 * Computes: Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 *
 * Performance Optimization:
 * 1. Register Caching: Frequently used pointers and loop variables are stored 
 *    in registers to minimize memory latency.
 * 2. Pointer Arithmetic: Replaces array indexing with direct increments 
 *    to reduce address calculation overhead.
 * 3. Cache Locality: Inner loops are structured to traverse memory linearly.
 *
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

#include "utils.h"

/**
 * my_solver - Optimized implementation using pointer arithmetic and register hints.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	register int i, j, k;

	/**
	 * Pre-condition: Allocation of result and intermediate matrices.
	 */
	double* C = (double*)calloc(N * N, sizeof(double));
	double* AB = (double*)calloc(N * N, sizeof(double)); 
	double* prod1 = (double*)calloc(N * N, sizeof(double)); 
	double* prod2 = (double*)calloc(N * N, sizeof(double)); 
	if (C == NULL || AB == NULL || prod1 == NULL || prod2 == NULL) {
        perror("calloc failed\n");
        exit(EXIT_FAILURE);
    }

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Uses a cached row pointer for A and linear increments 
	 * for column access in B.
	 */
	for (i = 0; i < N; i++) {
		register double* lineA = A + i * N + i;
		register double* pAB = AB + i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			register double* elemA = lineA;
			register double* columnB = B + i * N + j;
			for (k = i; k < N; k++) {
				sum += *elemA * *columnB;
				elemA++;
				columnB += N;
			}
			*pAB = sum;
			pAB++;
		}
	}

	/**
	 * Block Logic: Compute prod1 = AB * B^T.
	 * Optimization: Linear memory traversal for both operands by exploiting 
	 * row-major storage.
	 */
	for (i = 0; i < N; i++) {
		register double* lineAB = AB + i * N;
		register double* pProd1 = prod1 + i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			register double* elemAB = lineAB;
			register double* elemBt = B + j * N;
			for (k = 0; k < N; k++) {
				sum += *elemAB * *elemBt;
				elemAB++;
				elemBt++;
			}
			*pProd1 = sum;
			pProd1++;
		}
	}

	/**
	 * Block Logic: Compute prod2 = A^T * A.
	 * Optimization: Strided pointer traversal for the A^T (column) component.
	 */
	for (i = 0; i < N; i++) {
		register double* columnAt = A + i;
		register double* pProd2 = prod2 + i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			register double* elemAt = columnAt;
			register double* elemA = A + j;
			for (k = 0; k < N; k++) {
				sum += *elemAt * *elemA;
				elemA += N;
				elemAt += N;
			}
			*pProd2 = sum;
			pProd2++;
		}
	}

	/**
	 * Block Logic: Final summation.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = prod1[i * N + j] + prod2[i * N + j];
		}
	}

	free(AB);
	free(prod1);
	free(prod2);
	return C;	
}
