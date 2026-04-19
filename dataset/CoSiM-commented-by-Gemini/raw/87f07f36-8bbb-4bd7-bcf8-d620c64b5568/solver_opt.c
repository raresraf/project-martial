/**
 * @raw/87f07f36-8bbb-4bd7-bcf8-d620c64b5568/solver_opt.c
 * @brief Optimized dense matrix multiplication solver computing C = A * B * B^T + A^T * A.
 * Algorithm: Loop unrolling, pointer arithmetic, and constant caching tailored for scalar accumulation.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ tracking internal partial product evaluations.
 */
#include "utils.h"

/**
 * Functional Utility: Optimizes upper triangular partial product C = A * B.
 */
void matrix_multiplication_with_superior(int N, double *A, double *B, double *C) {	
	int i, j, k;

	/**
	 * Block Logic: Evaluates the upper triangular sections applying pointer caching to matrix rows.
	 * Invariant: Avoids recalculating base row offsets `i * N` within the innermost loops.
	 */
	for (i = 0; i < N; i++) {
		double *pa = &A[i * N];
		double *pc = C + i * N;

		for (j = 0; j < N; j++) {
			register double x = 0;
			
			for (k = i; k < N; k++) {
				x += *(pa + k) * B[k * N + j];
			}
			*(pc + j) = x;
		}
	}
}

/**
 * Functional Utility: Performs dot products representing C = A * B^T utilizing pointers.
 */
void matrix_multiplication_with_transpose(int N, double *A, double *B, double *C) {	
	int i, j, k;
	double *temp = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Aggregates row values leveraging localized pointer strides.
	 * Invariant: Exploits sequential access on row arrays mitigating array index arithmetic overhead.
	 */
	for (i = 0; i < N; i++) {
		double *pa = A + i * N;
		double *ptemp = temp + i * N;		

		for (j = 0; j < N; j++) {
			double *pb = B + j * N;
			register double x = 0;

			for (k = 0; k < N; k++) {
				x += *(pa + k) * *(pb + k);
			}

			*(ptemp + j) = x;
		}
	}

	for (i = 0; i < N; i++) {
		double *pc = C + i * N;
		double *ptemp = temp + i * N;		

		for (j = 0; j < N; j++) {
			*(pc + j) = *(ptemp + j);
		}
	}

	free(temp);
}

/**
 * Functional Utility: Employs cached pointers to rapidly compute and add A^T * A.
 */
void matrix_multiplication_with_lower_upper(int N, double *A, double *C) {
	double *temp = calloc(N * N, sizeof(double));
	int i, j, k; 

	/**
	 * Block Logic: Minimizes iterations by limiting upper bounds mathematically dynamically.
	 * Invariant: The index constraint `del = min(i, j)` bypasses explicit zero multiplication.
	 */
	for (i = 0; i < N; i++) {
		double *ptemp = temp + i * N;
		
		for (j = 0; j < N; j++) {
			register double x = 0;

			register int del = (i < j) ? i : j;

			for (k = 0; k <= del; k++) {
			 	x += A[k * N + i] * A[k * N + j];
			}

			*(ptemp + j) = x;
		}
	}

	for (i = 0; i < N; i++) {
		double *pc = C + i * N;
		double *ptemp = temp + i * N;

		for (j = 0; j < N; j++) {
			*(pc + j) += *(ptemp + j);
		}
	}

	free(temp);
}

void print_matrix(int N, double *a) {
	int i, j;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			printf("%f ", a[i + j]);
		}
		printf("\n");
	}
}

/**
 * Functional Utility: Orchestrates the optimized computations sequentially.
 */
double* my_solver(int N, double *A, double* B) {
	double *C = calloc(N * N, sizeof(double));
	
	matrix_multiplication_with_superior(N, A, B, C);
	matrix_multiplication_with_transpose(N, C, B, C);
	matrix_multiplication_with_lower_upper(N, A, C);

	return C;
}
