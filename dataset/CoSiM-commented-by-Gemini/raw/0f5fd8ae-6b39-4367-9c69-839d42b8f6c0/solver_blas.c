/**
 * @file solver_blas.c
 * @brief A modular matrix solver implementation using a BLAS library (CBLAS interface).
 *
 * This file provides an implementation of the `my_solver` function that computes
 * the expression `C = (A * B) * B^T + A^T * A`. The implementation is broken down
 * into several helper functions, each responsible for a specific matrix multiplication
 * step, which is then performed by a call to a BLAS function.
 */
#include <string.h>

#include "utils.h"
#include "cblas.h"


/**
 * @brief Computes the matrix product AB = A * B.
 *
 * It uses the triangular matrix multiplication function `cblas_dtrmm`, assuming
 * A is an upper triangular matrix.
 *
 * @param N The dimension of the matrices.
 * @param A A pointer to the N x N input matrix A.
 * @param B A pointer to the N x N input matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the result A * B.
 */
double* AB(int N, double *A, double *B) {
	double *AB;
	
	AB = calloc(N * N, sizeof(double));
	if (!AB) {
		exit(EXIT_FAILURE);
	}

	
	memcpy(AB, B, N * N * sizeof(double));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N, 1.0, A, N,
		AB, N);

	return AB;
}


/**
 * @brief Computes the matrix product A_tA = A^T * A.
 *
 * It uses the triangular matrix multiplication function `cblas_dtrmm` with the
 * `CblasTrans` flag to perform the multiplication with the transpose of A.
 *
 * @param N The dimension of the matrices.
 * @param A_t This parameter is unused.
 * @param A A pointer to the N x N input matrix A.
 * @return A pointer to a newly allocated N x N matrix containing the result A^T * A.
 */
double* A_tA(int N, double *A_t, double *A) {
	double *A_tA;
	
	A_tA = calloc(N * N, sizeof(double));
	if (!A_tA) {
		exit(EXIT_FAILURE);
	}

	
	memcpy(A_tA, A, N * N * sizeof(double));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N, 1.0, A, N,
		A_tA, N);

	return A_tA;
}


/**
 * @brief Computes the final sum: (A * B) * B^T + (A^T * A).
 *
 * This function takes the results of the previous multiplication steps and
 * performs the final computation. It uses `cblas_dgemm` to calculate `(A * B) * B^T`
 * and add it to the `(A^T * A)` matrix, which is passed in and pre-loaded into
 * the destination matrix `sum`.
 *
 * @param N The dimension of the matrices.
 * @param AB Pointer to the pre-computed (A * B) matrix.
 * @param B Pointer to the input matrix B.
 * @param A_tA Pointer to the pre-computed (A^T * A) matrix.
 * @return A pointer to a newly allocated N x N matrix containing the final result.
 */
double* ABB_t(int N, double *AB, double *B, double *A_tA) {
	double *sum;
	
	sum = calloc(N * N, sizeof(double));
	if (!sum) {
		exit(EXIT_FAILURE);
	}

	
	memcpy(sum, A_tA, N * N * sizeof(double));

	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N, 1.0, AB,
		N, B, N, 1.0, sum, N);

	return sum;
}

/**
 * @brief Computes the expression C = (A * B) * B^T + A^T * A by calling modular BLAS helper functions.
 *
 * This function orchestrates the matrix computation by calling a sequence of helper
 * functions, each of which wraps a specific BLAS call.
 *
 * @param N The dimension of the square matrices.
 * @param A A pointer to the N x N input matrix A.
 * @param B A pointer to the N x N input matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the final result.
 *
 * @note The sequence of operations is:
 * 1. `ptr_AB = A * B`
 * 2. `ptr_A_tA = A^T * A`
 * 3. `ptr_ABB_t = (ptr_AB * B^T) + ptr_A_tA`
 * 4. Frees intermediate results and returns the final matrix.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER
");
	double *ptr_AB = AB(N, A, B);
	double *ptr_A_tA = A_tA(N, A, A);
	double *ptr_ABB_t = ABB_t(N, ptr_AB, B, ptr_A_tA);

	free(ptr_AB);
	free(ptr_A_tA);

	return ptr_ABB_t;
}
