/**
 * @file solver_blas.c
 * @brief Implements a matrix solver using BLAS (Basic Linear Algebra Subprograms) routines.
 * This file provides a high-performance solution for computing a specific matrix expression:
 * `C = (A * B) * B^T + A^T * A`. It leverages optimized CBLAS functions for operations on
 * double-precision floating-point matrices.
 * Algorithm:
 *   1. Initializes an intermediate matrix `AB` as a copy of `B`.
 *   2. Computes `AB = A * B` (triangular matrix multiplication, where `A` is upper triangular).
 *   3. Initializes an intermediate matrix `D` to zero. Computes `D = AB * B^T` (general matrix multiplication).
 *   4. Initializes an intermediate matrix `E` as a copy of `A`.
 *   5. Computes `E = A^T * A` (triangular matrix multiplication, where `A` is upper triangular and then transposed).
 *   6. Computes the final result `C = D + E` (element-wise addition).
 * Optimization: Utilizes highly optimized, hardware-accelerated BLAS/CBLAS libraries for all matrix operations,
 * ensuring peak performance for dense linear algebra computations.
 * Time Complexity: Dominated by matrix multiplications, theoretically $O(N^3)$ for $N \times N$ matrices,
 * but with significant constant factor improvements due to BLAS optimizations.
 * Space Complexity: $O(N^2)$ for storing the final result matrix `C` and intermediate matrices `AB`, `D`, and `E`.
 */

#include <string.h>
#include <stdlib.h>

#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	// Block Logic: Allocate memory for the final result matrix 'C'.
	double *C = calloc(N * N, sizeof(double));
	// Block Logic: Allocate memory for intermediate matrix `AB` (A * B).
	double *AB = calloc(N * N, sizeof(double));
	// Block Logic: Allocate memory for intermediate matrix `D` ((A * B) * B^T).
	double *D = calloc(N * N, sizeof(double));
	// Block Logic: Allocate memory for intermediate matrix `E` (A^T * A).
	double *E = calloc(N * N, sizeof(double));

	int i;

	// Conditional Logic: Handle memory allocation failure for any of the matrices.
	if(C == NULL || AB == NULL || D == NULL || E == NULL) {
		printf("Eroare la alocare\n");
		return NULL;
	}

	// Functional Utility: Copy matrix B into intermediate matrix AB.
	memcpy(AB, B, N * N * sizeof(double));

	// Functional Utility: Perform triangular matrix-matrix multiplication: AB = A * AB (effectively A * B).
	// CblasRowMajor: Matrices are stored in row-major order.
	// CblasLeft: A is on the left side of the multiplication.
	// CblasUpper: A is an upper triangular matrix.
	// CblasNoTrans: A is not transposed.
	// CblasNonUnit: A has non-unit diagonal elements.
	// 1.0: Scaling factor for the result.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
		N, N, 1.0, A, N, AB, N);

	// Functional Utility: Perform general matrix-matrix multiplication: D = AB * B^T.
	// CblasRowMajor: Matrices are stored in row-major order.
	// CblasNoTrans: AB is not transposed.
	// CblasTrans: B is transposed (B^T).
	// 1.0: Scaling factor for AB * B^T.
	// 0.0: Scaling factor for D (effectively overwriting D).
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N,
		1.0, AB, N, B, N, 0.0, D, N);

	// Functional Utility: Copy matrix A into intermediate matrix E.
	memcpy(E, A, N * N * sizeof(double));

	// Functional Utility: Perform triangular matrix-matrix multiplication: E = A^T * E (effectively A^T * A).
	// CblasRowMajor: Matrices are stored in row-major order.
	// CblasLeft: A is on the left side of the multiplication.
	// CblasUpper: A is an upper triangular matrix.
	// CblasTrans: A is transposed (A^T).
	// CblasNonUnit: A has non-unit diagonal elements.
	// 1.0: Scaling factor for the result.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
		N, N, 1.0, A, N, E, N);

	// Block Logic: Perform element-wise addition C = D + E.
	// This loop iterates through all elements, summing the corresponding values from D and E.
	for(i = 0; i < N * N; i++) {
		C[i] = D[i] + E[i];
	}

	// Block Logic: Free memory allocated for intermediate matrices.
	free(AB);
	free(D);
	free(E);

	// Functional Utility: Return the final computed result matrix C.
	return C;
}
