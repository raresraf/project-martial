
/**
 * @file solver_blas.c
 * @brief A matrix solver implementation using BLAS routines.
 * @note This version is memory-efficient, using a single auxiliary buffer (`aux`)
 *       to store several different intermediate results throughout the calculation.
 */
#include "utils.h"
#include <cblas.h>


/**
 * @brief Performs a sequence of matrix operations using BLAS.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix C. The caller is responsible for freeing this memory.
 *
 * @note The function computes the expression: C = (A * B) * B^T + A^T * A.
 *       It uses a single auxiliary buffer for multiple steps to save memory:
 *       1. `aux` stores B.
 *       2. `aux` is updated to A * B.
 *       3. The first term (A*B)*B^T is computed and stored in C.
 *       4. `aux` is overwritten with A.
 *       5. `aux` is updated to A^T * A.
 *       6. The second term is added to C using `daxpy`.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER
");
	// Allocate memory for the final result and one auxiliary matrix.
	double *C = malloc(N * N * sizeof(*C));
	double *aux = malloc(N * N * sizeof(*aux));

	
	// Step 1: Copy B into the auxiliary buffer. aux = B.
	cblas_dcopy(N * N, B, 1, aux, 1);

	
	// Step 2: Compute aux = A * aux, which is A * B. A is treated as upper triangular.
	// cblas_dtrmm: Triangular Matrix-Matrix Multiply.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, aux, N);

	
	// Step 3: Compute C = aux * B^T, which is (A * B) * B^T.
	// cblas_dgemm: General Matrix-Matrix Multiply.
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, aux, N, B, N, 0.0, C, N);

	
	// Step 4: Overwrite aux with A. aux = A.
	cblas_dcopy(N * N, A, 1, aux, 1);

	
	// Step 5: Compute aux = A^T * aux, which is A^T * A.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, aux, N);

	
	// Step 6: Add the two terms. C = aux + C.
	// cblas_daxpy performs the vector operation Y = a*X + Y.
	cblas_daxpy(N * N, 1, aux, 1, C, 1);

	// Free the auxiliary buffer and return the result.
	free(aux);
	return C;
}
