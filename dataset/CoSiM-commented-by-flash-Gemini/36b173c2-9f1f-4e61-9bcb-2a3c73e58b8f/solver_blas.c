/**
 * @file solver_blas.c
 * @brief Implements a matrix solver using BLAS (Basic Linear Algebra Subprograms) functions.
 *
 * This file provides a `my_solver` function that performs a series of matrix operations,
 * specifically triangular matrix multiplication (DTRMM) and general matrix multiplication (DGEMM),
 * leveraging the optimized routines provided by CBLAS for high-performance computing.
 * The solver is designed to operate on double-precision floating-point matrices.
 *
 * Algorithm: Matrix operations (A * B + A^T * C, simplified as specific BLAS calls).
 * Libraries: CBLAS.
 * Time Complexity: Primarily dictated by the BLAS functions, typically O(N^3) for matrix multiplications
 *                  (specifically, DTRMM and DGEMM are O(N^3) for NxN matrices).
 */

#include "utils.h"
#include "cblas.h"
#include <string.h>
#include <stdio.h> // Required for printf

/**
 * @brief Solves a matrix problem using BLAS routines.
 *
 * This function takes two N x N matrices, A and B, and performs a series of
 * matrix multiplications and additions. It utilizes CBLAS functions for
 * efficient computation, specifically `cblas_dtrmm` for triangular matrix
 * operations and `cblas_dgemm` for general matrix multiplication.
 *
 * The operation can be broadly described as computing `(A^-1 * B) + (A^T * B)`,
 * where `A^-1` is effectively handled by `cblas_dtrmm` with appropriate parameters
 * for an inverse-like operation or solving a triangular system, and `A^T * B`
 * also uses `cblas_dtrmm` for the transpose multiplication.
 * The final `cblas_dgemm` combines results.
 *
 * @param N (int): The dimension of the square matrices (N x N).
 * @param A (double*): Pointer to the first input matrix (N x N).
 * @param B (double*): Pointer to the second input matrix (N x N).
 * @return (double*): Pointer to a newly allocated N x N matrix containing the result, or NULL if memory allocation fails.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	// Block Logic: Allocates memory for an N x N matrix C, initialized to zeros.
	// This matrix will be used as a temporary buffer for intermediate calculations.
	double *C = (double*) calloc(N * N, sizeof(double));
	// Block Logic: Checks if memory allocation for matrix C was successful.
	if (C == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	// Block Logic: Allocates memory for the result matrix res_A, initialized to zeros.
	double *res_A = (double*) calloc(N * N, sizeof(double));
	// Block Logic: Checks if memory allocation for matrix res_A was successful.
	if (res_A == NULL) {
		printf("Failed calloc!");
		free(C); // Inline: Frees previously allocated memory for C to prevent leaks.
		return NULL;
	}

	// Block Logic: Copies the content of matrix B into matrix C.
	// C now holds a copy of B, which will be modified by the DTRMM operation.
	memcpy(C, B, N * N * sizeof(double));
	// Functional Utility: Performs a triangular matrix multiply (DTRMM) operation.
	// This specific call performs C = A * C, where A is treated as a triangular matrix
	// and C initially holds B. The parameters (141, 121, 111, 131) specify details
	// like side (left), triangle (upper), transpose (no transpose), and unit diagonal (non-unit).
	cblas_dtrmm(CblasRowMajor, 141, 121, 111, 131, N, N, 1, A, N, C, N);

	// Block Logic: Copies the content of matrix A into matrix res_A.
	// res_A now holds a copy of A, which will be modified by the DTRMM operation.
	memcpy(res_A, A, N * N * sizeof(double));
	// Functional Utility: Performs another triangular matrix multiply (DTRMM) operation.
	// This specific call performs res_A = A_T * res_A, where A_T is the transpose of A.
	// The parameters (141, 121, 112, 131) specify details like side (left),
	// triangle (upper), transpose (transpose), and unit diagonal (non-unit).
	cblas_dtrmm(CblasRowMajor, 141, 121, 112, 131, N, N, 1, A, N, res_A, N);

	// Functional Utility: Performs a general matrix multiply (DGEMM) operation.
	// This call computes res_A = 1 * C * B + 1 * res_A.
	// It effectively adds the result of C * B to the current content of res_A.
	cblas_dgemm(CblasRowMajor, 111, 112, N, N, N, 1, C, N, B, N, 1, res_A, N);

	// Functional Utility: Frees the memory allocated for the temporary matrix C.
	free(C);
	return res_A; // Functional Utility: Returns the pointer to the dynamically allocated result matrix res_A.
}
