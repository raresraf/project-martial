/**
 * @file solver_opt.c
 * @brief Manually optimized Implementation of the matrix expression solver.
 *
 * Computes: Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 *
 * Performance Optimization:
 * 1. Register Caching: Frequently accessed row pointers and accumulators 
 *    are stored in registers.
 * 2. Symmetry Exploitation: Only computes half of symmetric products 
 *    (B*B^T and A^T*A) and mirrors results.
 * 3. Pointer Arithmetic: Uses direct offset increments to reduce address 
 *    calculation latency.
 *
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

#include "utils.h"

int min(int a, int b) {
	return (a < b) ? a : b;
}

/**
 * my_solver - Optimized implementation using symmetry and register hints.
 */
double* my_solver(int N, double *A, double* B) {
	register int i, j, k;

	/**
	 * Pre-condition: Allocation of result and scratchpad buffers.
	 */
	double *BB_T = calloc(N * N, sizeof(double));
	if(BB_T == NULL) {
		perror("Calloc failed!");
		exit(ENOMEM);
	}

	double *C = calloc(N * N, sizeof(double));
	if(C == NULL) {
		perror("Calloc failed!");
		exit(ENOMEM);
	}

	/**
	 * Block Logic: Compute BB_T = B * B^T.
	 * Optimization: Exploits result symmetry; computes upper triangular part 
	 * and copies to lower part.
	 */
	for(i = 0; i < N; i++) {
		register double *orig_pa = &B[i * N];
		for(j = i; j < N; j++) {
			register double *pa = orig_pa;
			register double *pb = &B[j * N];
			register double sum = 0.0;
			for(k = 0; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb++;
			}
			BB_T[i * N + j] = sum;
			if(i != j)
				BB_T[j * N + i] = sum;
		}
	}

	/**
	 * Block Logic: Compute C = A * BB_T.
	 * Optimization: Caches row pointer of A and exploits upper triangularity.
	 */
	for(i = 0; i < N; i++) {
		register double *orig_pa = &A[i * N + i];
		for(j = 0; j < N; j++) {
			register double *pa = orig_pa;
			register double *pb = &BB_T[j + i * N];
			register double sum = 0.0;
			for(k = i; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb += N;
			}
			C[i * N + j] = sum; 
		}
	}

	/**
	 * Block Logic: Accumulate C += A^T * A.
	 * Optimization: Uses symmetry and triangularity to bound the inner loop 
	 * and mirrors the symmetric part.
	 */
	for(i = 0; i < N; i++) {
		register double *orig_pa = &A[i];
		for(j = i; j < N; j++) {
			register double *pa = orig_pa;
			register double *pb = &A[j];
			register double sum = 0.0;
			for(k = 0; k <= min(i, j); k++) {
				sum += (*pa) * (*pb);
				pa += N;
				pb += N;
			}
			C[i * N + j] += sum;
			if(i != j)
				C[j * N + i] += sum;
		}
	}

	free(BB_T);

	return C;	
}
