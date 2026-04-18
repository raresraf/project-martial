
/**
 * @file solver_opt.c
 * @brief An optimized C implementation of a matrix solver.
 *
 * This file contains a version of the matrix solver that includes several
 * manual optimizations compared to the naive implementation. It computes the
 * same mathematical expression but rearranges the order of operations and
 * exploits properties of the intermediate matrices.
 */

#include "utils.h"


/**
 * @brief Performs a sequence of matrix operations using optimized loops.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix F. The caller is responsible for freeing this memory.
 *
 * @note This function computes the expression: F = A * (B * B^T) + A^T * A.
 *       This is mathematically equivalent to the other solver versions.
 *
 *       Optimizations applied:
 *       1.  **Reordering of Operations:** It first computes C = B * B^T, and then D = A * C,
 *           which can have different performance characteristics than (A * B) * B^T.
 *       2.  **Symmetry Exploitation:** The intermediate matrices C = B * B^T and E = A^T * A are symmetric.
 *           The code calculates only the lower (or upper) triangle and then copies the values,
 *           reducing the number of computations by almost half for these steps.
 *       3.  **Pointer Arithmetic:** It uses pointers to iterate through the matrix rows,
 *           which can sometimes be optimized better by the compiler than array-style indexing.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER
");
	int i, j, k;

	
	// Block Logic: Compute C = B * B^T.
	double *C = calloc(N * N, sizeof(double));
	
	// Optimization: Since B * B^T is symmetric, we only compute the lower triangle
	// and then copy the values to the upper triangle.
	for (i = 0; i < N; i++) {
		double* ptrc_i = C + i * N;
		double* ptrb_i = B + i * N;
		for (j = 0; j <= i; j++) {
			double* ptrc_j = C + j * N;
			double* ptrb_j = B + j * N;
			double temp = 0.0;
			// Computes the dot product of row i and row j of B.
			for (k = 0; k < N; k++) {
				temp += *(ptrb_i + k) * *(ptrb_j + k);
			}
			// C[i][j] = C[j][i] = temp
			*(ptrc_i + j) = temp;
			*(ptrc_j + i) = temp;
		}
	}

	
	// Block Logic: Compute D = A * C, where A is treated as an upper triangular matrix.
	double *D = calloc(N * N, sizeof(double));
	
	for (i = 0; i < N; i++) {
		double* ptrd_i = D + i * N;
		double* ptra_i = A + i * N;
		// The loop for k starts from i, assuming A is upper triangular.
		for (k = i; k < N; k++) {
			
			
			double* ptrc_k = C + k * N;
			for (j = 0; j < N; j++) {
				*(ptrd_i + j) += *(ptra_i + k) * *(ptrc_k + j);
			}
		}
	}

	
	// Block Logic: Compute E = A^T * A.
	double *E = calloc(N * N, sizeof(double));
	
	// Optimization: Similar to C, E is symmetric, so we compute one triangle
	// and copy the values.
	for (i = 0; i < N; i++) {
		double* ptre_i = E + i * N;
		double* ptra_i = A + i;
		
		for (j = 0; j <= i; j++) {
			int min = i < j ? i : j;
			double temp = 0;
			double* ptre_j = E + j * N;
			double* ptra_j = A + j;
			
			// Computes the dot product of column i and column j of A.
			for (k = 0; k <= min; k++) {
				temp += *(ptra_i + k * N) * *(ptra_j + k * N);
			}
			// E[i][j] = E[j][i] = temp
			*(ptre_i + j) = temp;
			*(ptre_j + i) = temp;
		}
	}
	
	// Allocate memory for the final result matrix F.
	double *F = calloc(N * N, sizeof(double));
	
	// Block Logic: Perform the final addition F = D + E.
	for(i = 0; i < N; i++) {
		double* ptrf_i = F + i * N;
		double* ptrd_i = D + i * N;
		double* ptre_i = E + i * N;
        for(j = 0; j < N; j++) {
			*(ptrf_i + j) = *(ptrd_i + j) + *(ptre_i + j);
        }
    }

	// Free the memory used for intermediate matrices.
	free(C);
	free(D);
	free(E);
	return F;
	
}
