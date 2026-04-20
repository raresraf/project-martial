/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */

#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "cblas.h"


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {

	double *C;
	double *AB;

	
	C = malloc(N * N * sizeof(*C));
	AB = malloc(N * N * sizeof(*AB));

	
	memcpy(C, A, N * N * sizeof(*C));
	memcpy(AB, B, N * N * sizeof(*AB));

	
	cblas_dtrmm( CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasTrans,
				CblasNonUnit,
				N, N, 1.0,
				A, N,
				C, N);

	
	cblas_dtrmm( CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasNoTrans,
				CblasNonUnit,
				N, N, 1.0,
				A, N,
				AB, N);

	
	cblas_dgemm( CblasRowMajor, 
				CblasNoTrans,
				CblasTrans,
				N, N, N, 1.0,
				AB, N,
				B, N,
				1.0, C, N);


	free(AB);

	printf("BLAS SOLVER\n");

	return C;
}
