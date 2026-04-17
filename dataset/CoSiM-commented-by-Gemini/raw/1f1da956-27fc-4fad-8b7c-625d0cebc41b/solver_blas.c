/**
 * @file solver_blas.c
 * @brief A BLAS-based implementation of a matrix solver.
 * @details This file provides a high-performance solution for the matrix equation
 * C = (A * B) * B' + A' * A, where A is an upper triangular matrix. It leverages the
 * Basic Linear Algebra Subprograms (BLAS) library for optimized matrix operations.
 */
#include <string.h>
#include "utils.h"
#include "cblas.h"

/**
 * @brief Performs element-wise addition of two matrices, C = A + B.
 * @details This version uses pointer arithmetic for sequential memory access.
 * Time Complexity: O(N^2)
 */
void addition(double *C, double *A, double *B, int N) {
	for (register int i = 0; i < N; ++i) {
		for (register int j = 0; j < N; ++j) {
			C[i * N + j] = *A + *B;
			++A;
			++B;
		}
	}
}

/**
 * @brief Solves C = (A * B) * B' + A' * A using BLAS functions.
 * @param N The dimension of the square matrices.
 * @param A A pointer to the input upper triangular matrix A (N x N).
 * @param B A pointer to the input matrix B (N x N).
 * @return A pointer to the resulting matrix C (N x N).
 *
 * @details The function uses highly optimized BLAS routines to perform the calculation:
 * 1. An intermediate matrix `AB` is created as a copy of `B`.
 * 2. `cblas_dtrmm` (triangular matrix multiply) calculates `AB = A * AB`, so `AB` becomes `A * B`.
 * 3. `cblas_dgemm` calculates `ABBt = AB * B'`, which is `(A * B) * B'`.
 * 4. An intermediate matrix `AtA` is created as a copy of `A`.
 * 5. `cblas_dtrmm` calculates `AtA = A' * AtA`, so `AtA` becomes `A' * A`.
 * 6. A final addition function sums the two resulting terms into `C`.
 */
double* my_solver(int N, double *A, double* B) {
	// Allocate memory for the final result matrix C.
	double *C = calloc(N * N, sizeof(*C));
	if (!C) {
		exit(-1);
	}

	// Allocate and initialize AB = B.
	double *AB = malloc(N * N * sizeof(*AB));
	if (!AB) {
		exit(-1);
	}
	memcpy(AB, B, N * N * sizeof(*B));

	// Allocate memory for the first term: (A * B) * B'
	double *ABBt = calloc(N * N, sizeof(*ABBt));
	if (!ABBt) {
		exit(-1);
	}

	// Allocate and initialize AtA = A.
	double *AtA = malloc(N * N * sizeof(*AtA));
	if (!AtA) {
		exit(-1);
	}
	memcpy(AtA, A, N * N * sizeof(*A));

	
	/**
	 * Block Logic: Step 1: Compute the intermediate product AB = A * B.
	 * `cblas_dtrmm` calculates `AB = alpha * op(A) * AB`, where op(A) is the
	 * upper triangular matrix A. The result overwrites the `AB` matrix.
	 */
	cblas_dtrmm(
		CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
		N, N, 1.0, A, N, AB, N
	);

	
	/**
	 * Block Logic: Step 2: Compute the first term ABBt = (A * B) * B'.
	 * `cblas_dgemm` calculates `ABBt = alpha * AB * B' + beta * ABBt`.
	 * With alpha=1.0 and beta=0.0, this computes `ABBt = (A * B) * B'`.
	 */
	cblas_dgemm(
		CblasRowMajor, CblasNoTrans, CblasTrans,
		N, N, N, 1.0, AB, N, B, N, 0.0, ABBt, N
	);

	
	/**
	 * Block Logic: Step 3: Compute the second term AtA = A' * A.
	 * `cblas_dtrmm` calculates `AtA = alpha * op(A) * AtA`, where op(A) is A'.
	 * The result overwrites the `AtA` matrix.
	 */
	cblas_dtrmm(
		CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
		N, N, 1.0, A, N, AtA, N
	);

	/**
	 * Block Logic: Step 4: Compute C = ABBt + AtA.
	 * This is an element-wise sum of the two terms.
	 */
	addition(C, ABBt, AtA, N);

	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
