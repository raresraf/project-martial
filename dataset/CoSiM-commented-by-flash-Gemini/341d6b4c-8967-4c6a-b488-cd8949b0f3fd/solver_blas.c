/**
 * @file solver_blas.c
 * @brief This file implements a matrix solver using the BLAS (Basic Linear Algebra Subprograms) library.
 * Specifically, it performs matrix operations for solving a linear system or matrix equation,
 * leveraging highly optimized routines provided by BLAS for performance.
 *
 * Algorithm: Matrix multiplication and triangular matrix multiplication using BLAS functions.
 * Time Complexity: Dominated by matrix multiplications, typically O(N^3) for N x N matrices.
 * Space Complexity: O(N^2) for storing intermediate matrices.
 */

#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "cblas.h"

/**
 * @brief Solves a matrix equation using BLAS (Basic Linear Algebra Subprograms) routines.
 * This function performs a sequence of matrix operations, including copying, triangular matrix
 * multiplication (`dtrmm`), and general matrix multiplication (`dgemm`), to compute a result matrix `C`.
 * The specific mathematical operation being solved depends on the combination of `dtrmm` and `dgemm` calls.
 *
 * @param N The dimension of the square matrices (N x N).
 * @param A A pointer to the input matrix A (N x N).
 * @param B A pointer to the input matrix B (N x N).
 * @return A pointer to the newly allocated result matrix C (N x N). The caller is responsible for freeing this memory.
 */
double* my_solver(int N, double *A, double *B) {
	// Functional Utility: Indicate the usage of the BLAS solver to the standard output for tracing or debugging.
	printf("BLAS SOLVER\n");
	// Functional Utility: Allocate memory for result matrix C and intermediate matrix AB, initializing to zeros.
	double *C = calloc(N * N, sizeof(double));
	double *AB = calloc(N * N, sizeof(double));

	// Functional Utility: Initialize intermediate matrix AB with the contents of matrix B.
	// Precondition: `AB` and `B` point to valid memory blocks of size `N * N * sizeof(double)`.
	memcpy(AB, B, N * N * sizeof(double));
	
	// Functional Utility: Initialize result matrix C with the contents of matrix A.
	// Precondition: `C` and `A` point to valid memory blocks of size `N * N * sizeof(double)`.
	memcpy(C, A, N * N * sizeof(double));	

	// Block Logic: Perform triangular matrix multiplication: C = A_transposed_upper * C.
	// This operation effectively pre-multiplies C by the transpose of the upper triangular part of A.
	// Precondition: A and C are N x N matrices. `A` is treated as an upper triangular matrix and transposed.
	// Invariant: `C` holds the result of the triangular matrix multiplication.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, C, N);
	
	// Block Logic: Perform triangular matrix multiplication: AB = A_upper * AB.
	// This operation effectively pre-multiplies AB by the upper triangular part of A.
	// Precondition: A and AB are N x N matrices. `A` is treated as an upper triangular matrix.
	// Invariant: `AB` holds the result of the triangular matrix multiplication.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, AB, N);
	
	// Block Logic: Perform general matrix multiplication and accumulation: C = 1.0 * AB * B_transposed + 1.0 * C.
	// This operation updates C by adding the product of AB and the transpose of B to its current value.
	// Precondition: AB, B, and C are N x N matrices.
	// Invariant: `C` contains the final computed result after the combined matrix operations.
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, C, N);

	// Functional Utility: Release the dynamically allocated memory for the intermediate matrix `AB`.
	free(AB);
	
	return C;
}
