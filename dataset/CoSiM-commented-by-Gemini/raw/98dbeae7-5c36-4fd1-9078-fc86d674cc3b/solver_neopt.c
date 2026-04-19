/**
 * @raw/98dbeae7-5c36-4fd1-9078-fc86d674cc3b/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver computing C = A^T * A + A * (B * B^T).
 * Algorithm: Standard nested loops traversing the matrices generating element-wise answers.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ for intermediate target buffers.
 */

#include "utils.h"
#include "string.h"

/**
 * Functional Utility: Cumulates array `b` into array `c` simulating element-wise vector addition.
 */
void sum(double *c, double *b, int N) {
	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			c[i * N + j] += b[i * N + j];
		}
	}
}

/**
 * Block Logic: Solves c = a^T * b. Since a == b == A, it computes A^T * A.
 * Invariant: Limits internal multiplicative limits `k <= (i > j ? j : i)` based on symmetric upper triangular constraints.
 */
void mulAtA(double *c, double *a, double *b, int N) {
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k <= (i > j ? j : i); k++) {
				c[i * N + j] += a[k * N + i] * b[k * N + j];
			}
		}
	}
}

/**
 * Block Logic: Evaluates B * B^T utilizing symmetric row by row multiplication.
 */
void mulBBt(double *c, double *a, double *b, int N) {
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				c[i * N + j] += a[i * N + k] * b[j * N + k];
			}
		}
	}
}

/**
 * Block Logic: Generates c = a * b. Solves A * (B * B^T).
 * Invariant: Adheres to the upper-triangular structure of A by starting the third loop from `i`.
 */
void mulABBt(double *c, double *a, double *b, int N) {
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				c[i * N + j] += a[i * N + k] * b[k * N + j];
			}
		}
	}
}

/**
 * Functional Utility: Orchestrates the calculation sequence and orchestrates temporary memory allocation.
 */
double* my_solver(int N, double *A, double* B) {
	double *C = (double*)calloc(N * N, sizeof(double));
	double *D = (double*)calloc(N * N, sizeof(double));
	double *E = (double*)calloc(N * N, sizeof(double));
	
	mulAtA(C, A, A, N);
	
	mulBBt(D, B, B, N);
	
	mulABBt(E, A, D, N);
	
	sum(C, E, N);
	free(D); free(E);
	return C;
}
