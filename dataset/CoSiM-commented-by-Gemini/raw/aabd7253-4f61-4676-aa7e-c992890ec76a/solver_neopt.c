/**
 * @file solver_neopt.c
 * @brief A non-optimized, baseline implementation of a matrix expression solver.
 *
 * This file provides a straightforward implementation of a function that
 * calculates the matrix expression: (A_upper * B * B^T) + (A_lower^T * A_upper),
 * where A_upper and A_lower indicate that the matrix A is treated as upper or
 * lower triangular in specific multiplication steps. The implementation uses
 * naive O(N^3) matrix multiplication.
 */

#include "utils.h"

// Constants to define matrix type for multiplication
#define UPPER 1
#define LOWER -1
#define NORMAL 0


/**
 * @brief Transposes a square matrix.
 * @param N The dimension of the matrix.
 * @param C A pointer to the output matrix where the transpose will be stored.
 *          This is expected to be pre-allocated.
 * @param M The input matrix to be transposed.
 *
 * Time Complexity: O(N^2)
 */
void transpose(int N, double **C, double *M) {
	for(int i = 0; i < N; i ++) {
		for(int j = 0; j < N; j++) {
			(*C)[i * N + j] = M[j * N + i];
		}
	}
}

/**
 * @brief Performs matrix multiplication C = A * B with optional triangular properties.
 * @param N The dimension of the matrices.
 * @param C A pointer to the output matrix C. It is assumed to be initialized (e.g., with zeros).
 * @param A The first input matrix.
 * @param B The second input matrix.
 * @param typeA Specifies if matrix A is upper triangular (UPPER), lower triangular (LOWER), or normal (NORMAL).
 * @param typeB Specifies if matrix B has special properties (currently only NORMAL is fully handled in combination).
 *
 * Time Complexity: O(N^3)
 *
 * Algorithm: This is a naive triple-loop matrix multiplication. The `typeA` and `typeB`
 * parameters introduce branches to skip calculations if matrices are assumed to be
 * triangular, which can reduce operations but does not change the overall cubic complexity.
 */
void multiply(int N, double **C, double *A, double *B, int typeA, int typeB) {
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			for(int k = 0; k < N; k++) {
				if(typeA == NORMAL && typeB == NORMAL)
					(*C)[i * N + j] += A[i * N + k]* B[k * N + j];
				// Block Logic: If A is upper triangular, only compute for k >= i.
				else if (typeA == UPPER && typeB == NORMAL) {
					if(i <= k)
						(*C)[i * N + j] += A[i * N + k]* B[k * N + j];
				// Block Logic: If A is lower triangular and B is upper, apply both constraints.
				} else if (typeA == LOWER && typeB == UPPER) {
					if(i >= k && k <= j)
						(*C)[i * N + j] += A[i * N + k]* B[k * N + j];
				}
			}
		}
	}
}

/**
 * @brief Performs element-wise addition of two matrices, C = A + B.
 * @param N The dimension of the matrices.
 * @param C A pointer to the output matrix.
 * @param A The first input matrix.
 * @param B The second input matrix.
 *
 * Time Complexity: O(N^2)
 */
void add(int N, double **C, double *A, double *B) {
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			(*C)[i * N + j] = A[i * N + j] + B[i * N + j];
		}
	}
}


/**
 * @brief Calculates the matrix expression (A * B * B^T) + (A^T * A).
 * @param N The dimension of the square matrices A and B.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return A pointer to the resulting matrix. The caller is responsible for freeing this memory.
 *
 * Algorithm: The function computes the expression in several distinct steps,
 * allocating memory for intermediate results.
 * 1. Computes AB = A * B, treating A as upper triangular.
 * 2. Computes the transpose of B.
 * 3. Computes ABBt = AB * B^T.
 * 4. Computes the transpose of A.
 * 5. Computes AtA = A^T * A, treating A^T as lower and A as upper triangular.
 * 6. Adds the results: result = ABBt + AtA.
 */
double* my_solver(int N, double *A, double* B) {
	// Step 1: Calculate AB = A * B (A is upper triangular)
	double *AB = calloc(N* N,sizeof(double));
	multiply(N, &AB, A, B, UPPER, NORMAL);

	// Step 2: Transpose B
	double *Btrans = malloc((N * N) * sizeof(double));
	transpose(N, &Btrans, B);

	// Step 3: Calculate ABBt = AB * B^T
	double *ABBt = calloc(N* N,sizeof(double));
	multiply(N, &ABBt, AB, Btrans, NORMAL, NORMAL);

	// Step 4: Transpose A
	double *Atrans = malloc((N * N) * sizeof(double));
	transpose(N, &Atrans, A);

	// Step 5: Calculate AtA = A^T * A (A^T is lower, A is upper)
	double *AtA = calloc(N* N,sizeof(double));
	multiply(N, &AtA, Atrans, A, LOWER, UPPER);

	// Step 6: Add the two intermediate results
	double *result = malloc((N * N) * sizeof(double));
	add(N, &result, ABBt, AtA);

	// Free all intermediate allocated memory
	free(AB);
	free(Btrans);
	free(ABBt);
	free(Atrans);
	free(AtA);

	return result;
}
