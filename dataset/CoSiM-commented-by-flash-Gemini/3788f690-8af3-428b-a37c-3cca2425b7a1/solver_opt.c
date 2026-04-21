
#include <string.h>
#include "utils.h"

/**
 * @file solver_opt.c
 * @brief Manually optimized matrix solver implementation.
 * 
 * Functional Intent: Computes the result of the matrix expression:
 * Result = (A * B * B^T) + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Employs several low-level C optimization techniques:
 * 1. Pointer Arithmetic: Reduces indexing overhead by using direct address increments.
 * 2. Register Hints: Sugests the compiler keep frequently used accumulators in CPU registers.
 * 3. Cache locality: Accesses arrays in row-major order where possible and caches 
 *    row/column pointers to minimize redundant base address calculations.
 * 
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

/**
 * my_solver - Optimized implementation using pointer-based iteration.
 */
double* my_solver(int N, double *A, double* B) {
	double *C = (double*) calloc(N * N, sizeof(double));
	if (!C)
		return NULL;

	double *aux = (double*) calloc(N * N, sizeof(double));
	if (!aux)
		return NULL;

	/**
	 * Block Logic: Compute aux = A * B.
	 * Optimization: Pointer-based dot product.
	 * 1. Caches the base address of A's current row.
	 * 2. Uses linear probing (pa++) for the triangular part of A.
	 * 3. Uses strided probing (pb += N) for the columns of B.
	 */
	int i, j, k;
	for (i = 0; i < N; i++) {
		double *orig_pa = &A[i * N];
		for (j = 0; j < N; j++) {
			register double *pa = orig_pa + i;
			register double *pb = &B[i * N + j];
			register double suma = 0;
			for (k = i; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			aux[i * N + j] = suma;
		}
	}

	/**
	 * Block Logic: Compute C = aux * B^T.
	 * Optimization: Row-major dot product for transposed access.
	 * Logic: By accessing both matrices via linear pointer increments (paux++, pb_t++), 
	 * the code effectively performs a dot product between aux[i] and B[j], 
	 * which is mathematically identical to aux[i] * B^T[j].
	 */
	for (i = 0; i < N; i++) {
		double *origin_aux = &aux[i * N];
		for (j = 0; j < N; j++) {
			register double suma = 0;
			register double *paux = origin_aux;
			register double *pb_t = &B[j *  N];
			for (k = 0; k < N; k++) {
				suma += *paux * *pb_t;
				paux++;
				pb_t ++;
			}
			C[i * N + j] = suma;
		}
	}

	/**
	 * Block Logic: Compute aux = A^T * A.
	 * Optimization: Strided pointer access to simulate transposition.
	 * Logic: Both pa_t and pa are incremented by N to traverse the columns 
	 * of A, representing rows of A^T and columns of A respectively.
	 */
	memset(aux, 0, N * N * sizeof(double));
	for (i = 0; i < N; i++) {
		double *origin_at = &A[i];
		for (j = 0; j < N; j++) {
			register double suma = 0;
			register double *pa_t = origin_at;
			register double *pa = &A[j];
			for (k = 0; k < i + 1; k++) {
				suma += *pa_t * *pa;
				pa_t+= N;
				pa += N;
			}
			aux[i * N + j] = suma;
		}
	}

	/**
	 * Block Logic: Final accumulation pass.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] += aux[i * N + j];
		}
	}

	free(aux);
	return C;
}
