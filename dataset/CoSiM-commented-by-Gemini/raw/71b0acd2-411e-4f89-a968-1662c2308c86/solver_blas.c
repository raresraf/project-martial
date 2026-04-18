
/**
 * @file solver_blas.c
 * @brief A matrix solver implementation using BLAS routines.
 * @note This version has a significant side effect: it modifies the input matrix A in-place.
 *       This is generally considered poor API design.
 */
#include "utils.h"
#include <string.h>
#include "cblas.h"


/**
 * @brief Allocates memory for a square matrix of size N x N.
 * @param N The dimension of the matrix.
 * @param C A pointer to the double pointer that will hold the allocated memory.
 */
void alloc_matrix(int N, double **C) {
		*C = malloc(N * N * sizeof(**C));
	if (NULL == *C)
		exit(EXIT_FAILURE);
}

/**
 * @brief Performs a sequence of matrix operations using BLAS.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N). This matrix is modified in-place.
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix C. The caller is responsible for freeing this memory.
 *
 * @warning This function has a major side effect: it overwrites the input matrix A
 *          with the result of A^T * A.
 *
 * @note The function computes the expression: C = A * (B * B^T) + A^T * A.
 *       The steps are:
 *       1. `cblas_dgemm` to compute B * B^T.
 *       2. `cblas_dtrmm` to compute A * (B * B^T).
 *       3. `cblas_dtrmm` to compute A^T * A (in-place, modifying A).
 *       4. A manual loop to perform the final matrix addition.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER
");

	double *C;
	int i;

	
	alloc_matrix(N, &C);

	
	// Step 1: Compute C = B * B^T.
	// cblas_dgemm: General Matrix-Matrix Multiply.
	cblas_dgemm(CblasRowMajor,
				CblasNoTrans,
				CblasTrans,
				N, N, N, 1,
				B, N,
				B, N,
				0.0, C, N);
	
	
	// Step 2: Compute C = A * C, which is now A * (B * B^T).
	// cblas_dtrmm: Triangular Matrix-Matrix Multiply, where A is upper triangular.
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1, A, N,
		C, N);

	
	// Step 3: Compute A = A^T * A.
	// WARNING: This operation is destructive and modifies the input matrix A in-place.
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1, A, N,
		A, N);

	
	// Step 4: Perform the final addition: C = C + A (where A is now A^T * A).
	for(i = 0; i < N * N; i++) {
			C[i] += A[i];
	}

	return C;
}
