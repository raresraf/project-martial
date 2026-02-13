/**
 * @file solver_neopt.c
 * @brief This file implements a non-optimized matrix solver, featuring basic matrix operations
 * such as addition, and specialized multiplications for normal-transpose and upper-triangular matrices.
 * This implementation serves as a baseline for performance comparisons, contrasting with optimized
 * versions like those utilizing BLAS.
 *
 * Algorithm: Naive matrix operations (addition, multiplication), no advanced optimizations (e.g., tiling, Strassen).
 * Time Complexity: Predominantly O(N^3) for matrix multiplications, and O(N^2) for addition, where N is matrix dimension.
 * Space Complexity: O(N^2) for storing intermediate matrices.
 */

#include "utils.h"




/**
 * @brief Performs element-wise addition of two matrices.
 * This function computes C = A + B for N x N matrices stored in row-major order.
 *
 * @param N The dimension of the square matrices.
 * @param a Pointer to the first input matrix A.
 * @param b Pointer to the second input matrix B.
 * @param c Pointer to the output matrix C, where the result will be stored.
 */
void add(int N, double *a, double *b, double *c) {
	int i;
	// Block Logic: Iterate through all elements of the matrices.
	// Precondition: Matrices a, b, and c are allocated for N*N doubles.
	// Invariant: After each iteration, c[i] contains the sum of a[i] and b[i].
	for (i = 0; i < N * N; i++) {
		c[i] = a[i] + b[i];
	}
}



void normal_x_normal_transpose(int N, double *a, double *c) {

	int i, j, k;

	for (i = 0; i < N; i++) {
		
		for (j = 0; j <= i; j++) {
			for (k = 0; k < N; k++) {
				
				c[i * N + j] += a[i * N + k] * a[j * N + k];
				
				c[j * N + i] = c[i * N + j];
			}
		}
	}
}



void upper_x_normal(int N, double *a, double *b, double *c) {

	int i, j, k;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			
			for (k = i; k < N; k++) {
				c[i * N + j] += a[i * N + k] * b[k * N + j];
			}
		}
	}
}


void upper_transpose_x_upper(int N, double *a, double *c) {

	int i, j, k;

	for (i = 0; i < N; i++) {
		
		for (j = 0; j <= i; j++) {
			
			for (k = 0; k <= j; k++) {
				
				c[i * N + j] += a[k * N + i] * a[k * N + j];
				c[j * N + i] = c[i * N + j];
			}
		}
	}
}

double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	double *C = calloc(N * N, sizeof(double));
	double *BBt = calloc(N * N, sizeof(double));
	double *ABBt = calloc(N * N, sizeof(double));
	double *AtA = calloc(N * N, sizeof(double));

	
	normal_x_normal_transpose(N, B, BBt);
	upper_x_normal(N, A, BBt, ABBt);
	upper_transpose_x_upper(N, A, AtA);
	add(N, ABBt, AtA, C);

	free(BBt);
	free(ABBt);
	free(AtA);

	return C;
}
