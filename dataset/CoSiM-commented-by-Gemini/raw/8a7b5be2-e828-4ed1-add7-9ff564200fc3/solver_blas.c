/**
 * @raw/8a7b5be2-e828-4ed1-add7-9ff564200fc3/solver_blas.c
 * @brief High-performance dense matrix solver using Level 3 BLAS operations.
 * Algorithm: Hardware-accelerated computation of C = (A * B) * B^T + A^T * A utilizing dtrmm and dgemm combinations.
 * Time Complexity: $O(N^3)$ (amortized via BLAS highly optimized kernels)
 * Space Complexity: $O(N^2)$ representing distinct output and intermediate buffers.
 */
#include <string.h>
#include <stdlib.h>
#include "cblas.h"
#include "utils.h"

double* my_solver(int N, double *A, double *B) {
	/**
	 * Functional Utility: Handles allocation of contiguous matrix buffers.
	 */
	double *AB;
	double *first;
	double *second;
	double *C;
	AB = calloc(N * N, sizeof(double));
	if(AB == NULL){
		exit(-1);
	}
	first = calloc(N * N, sizeof(double));
	if(first == NULL){
		exit(-1);
	}
	second = calloc(N * N, sizeof(double));
	if(second == NULL){
		exit(-1);
	}
	C = calloc(N * N, sizeof(double));
	if(C == NULL){
		exit(-1);
	}
    
	/**
	 * Functional Utility: Deep copies matrix B to AB to act as the source matrix for dtrmm.
	 */
	cblas_dcopy(N * N, B, 1.0, AB, 1.0);

	/**
	 * Functional Utility: Calculates the standard dot product AB = A * B in-place leveraging the upper triangular structure.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft,CblasUpper,CblasNoTrans,CblasNonUnit, N, N, 1.0, A, N, AB, N);
    
	/**
	 * Functional Utility: Calculates the trailing partial product first = AB * B^T.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, first, N);
    
	/**
	 * Functional Utility: Initializes second with A to prepare for the A^T multiplication step.
	 */
	cblas_dcopy(N * N, A, 1.0, second, 1.0);

	/**
	 * Functional Utility: Calculates second = A^T * A.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft,CblasUpper,CblasTrans,CblasNonUnit, N, N, 1.0, A, N, second, N);
    
	/**
	 * Functional Utility: Populates C with the matrix elements accumulated inside `first`.
	 */
	cblas_dcopy(N * N, first, 1.0, C, 1.0);

	/**
	 * Functional Utility: Finalizes C = C + second utilizing daxpy vector scaling addition.
	 */
	cblas_daxpy(N * N, 1.0, second, 1.0, C, 1.0);

	free(AB);
	free(first);
	free(second);

	return C;
}
