
/**
 * @file solver_neopt.c
 * @brief Naive (unoptimized) matrix solver implementation.
 * 
 * Functional Intent: Evaluates the matrix expression $Res = (A \times B) \times B^T + (A^T \times A)$ 
 * using basic nested-loop algorithms. It features explicit handling of 
 * triangular matrix constraints via conditional branches within the innermost 
 * loop, providing a baseline for performance comparison against BLAS or 
 * optimized manual implementations.
 * 
 * Domain: HPC, Algorithm Analysis, Matrix Computation.
 */

#include "utils.h"
#define UPPER 1
#define LOWER -1
#define NORMAL 0

/**
 * transpose - Computes the transpose of a square matrix.
 * 
 * Algorithm: Element-wise coordinate swap.
 * Time Complexity: $O(N^2)$
 */
void transpose(int N, double **C, double *M) {
	// Block Logic: Nested traversal for index permutation.
	for(int i = 0; i < N; i ++) {
		for(int j = 0; j < N; j++) {
			(*C)[i * N + j] = M[j * N + i];
		}
	}
}

/**
 * multiply - Performs general or constrained matrix-matrix multiplication.
 * 
 * Algorithm: Triple-nested loop dot product.
 * Logic: Supports normal, upper-triangular, and lower-triangular source constraints.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(1)$ additional workspace.
 */
void multiply(int N, double **C, double *A, double *B, int typeA, int typeB) {
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			for(int k = 0; k < N; k++) {
				if(typeA == NORMAL && typeB == NORMAL)
					(*C)[i * N + j] += A[i * N + k]* B[k * N + j];
				else if (typeA == UPPER && typeB == NORMAL) {
					// Logic: Optimization for upper-triangular A (skips elements where i > k).
					if(i <= k)
						(*C)[i * N + j] += A[i * N + k]* B[k * N + j];
				} else if (typeA == LOWER && typeB == UPPER) {
					// Logic: Optimization for product of Lower x Upper (dot product sub-range).
					if(i >= k && k <= j)
						(*C)[i * N + j] += A[i * N + k]* B[k * N + j];
				}		
			}
		}
	}
}

/**
 * add - Computes the element-wise sum of two matrices.
 * 
 * Time Complexity: $O(N^2)$
 */
void add(int N, double **C, double *A, double *B) {
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			(*C)[i * N + j] = A[i * N + j] + B[i * N + j];
		}
	}
}

/**
 * my_solver - Unoptimized orchestrator for the composite matrix expression.
 * 
 * Logic: Performs the sequence of operations (Multiply -> Transpose -> Multiply -> Add) 
 * using naive sub-routines.
 */
double* my_solver(int N, double *A, double *B) {
	double *AB = calloc(N* N,sizeof(double));
	multiply(N, &AB, A, B, UPPER, NORMAL);
	
	double *Btrans = malloc((N * N) * sizeof(double));
	transpose(N, &Btrans, B);
	
	double *ABBt = calloc(N* N,sizeof(double));
	multiply(N, &ABBt, AB, Btrans, NORMAL, NORMAL);

	double *Atrans = malloc((N * N) * sizeof(double));
	transpose(N, &Atrans, A);
	
	double *AtA = calloc(N* N,sizeof(double));
	multiply(N, &AtA, Atrans, A, LOWER, UPPER);

	double *result = malloc((N * N) * sizeof(double));
	add(N, &result, ABBt, AtA);

	free(AB);
	free(Btrans);
	free(ABBt);
	free(Atrans);
	free(AtA);

	return result;
}
