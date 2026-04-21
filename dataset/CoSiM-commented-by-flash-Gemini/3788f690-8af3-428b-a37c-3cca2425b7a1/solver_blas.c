
#include "utils.h"
#include <cblas.h>
#include <string.h>

/**
 * @file solver_blas.c
 * @brief High-performance matrix solver utilizing ATLAS/OpenBLAS routines.
 * 
 * Functional Intent: Computes the result of a specific matrix expression:
 * Result = (A * B * B^T) + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Leverages industry-standard BLAS implementations 
 * for level-3 matrix operations. It uses `cblas_dtrmm` to exploit the 
 * triangular property of matrix A and `cblas_dgemm` for efficient 
 * general matrix multiplication with transposition.
 * 
 * Domain: HPC, Linear Algebra, Performance Optimization.
 */

#define ALPHA 1.0

/**
 * my_solver - Implementation of the matrix expression via BLAS calls.
 * 
 * Algorithm: Multi-stage linear algebra transformation.
 * 1. Computes T = A * B using `cblas_dtrmm` (Left, Upper, NoTrans).
 * 2. Computes C = T * B^T using `cblas_dgemm` (NoTrans, Trans).
 * 3. Computes S = A^T * A using `cblas_dtrmm` (Left, Upper, Trans) on a copy of A.
 * 4. Aggregates the results (C = C + S).
 * 
 * Time Complexity: $O(N^3)$ dominated by matrix-matrix multiplications.
 */
double* my_solver(int N, double *A, double *B) {
	// Logic: Allocation and zero-initialization of the output matrix C.
	double *C = (double*) calloc(N * N, sizeof(double));
	if (!C)
		return NULL;

	// Logic: Temporary workspace for intermediate products.
	double *aux = (double*) calloc(N * N, sizeof(double));
	if (!aux)
		return NULL;

	// Block Logic: Calculation of (A * B * B^T).
	// Invariant: 'aux' initially stores B, then is transformed to (A * B).
	memcpy(aux, B, N * N * sizeof(double));
	
	// Step 1: aux = A * aux (where A is upper triangular).
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, ALPHA, A, N, aux, N);
	
	// Step 2: C = ALPHA * (aux * B^T) + ALPHA * C.
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, ALPHA, aux, N, B, N, ALPHA, C, N);

	// Block Logic: Calculation of (A^T * A).
	// Step 3: Transform a copy of A into (A^T * A).
	memcpy(aux, A, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, ALPHA, A, N, aux, N);

	// Block Logic: Final summation.
	// Logic: Naive element-wise addition to accumulate the symmetric product into the result.
	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] += aux[i * N + j];
		}
	}

	free(aux);
	return C;
}
