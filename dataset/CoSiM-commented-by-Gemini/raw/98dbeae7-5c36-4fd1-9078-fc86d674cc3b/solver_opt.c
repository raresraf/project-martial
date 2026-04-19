/**
 * @raw/98dbeae7-5c36-4fd1-9078-fc86d674cc3b/solver_opt.c
 * @brief Optimized dense matrix multiplication solver computing C = A^T * A + A * (B * B^T).
 * Algorithm: Implementation employing pointer iterations and cached transpose configurations.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ due to explicit copies and transpositions.
 */

#include "utils.h"
#include "string.h"

/**
 * Functional Utility: Accelerates element-wise matrix addition utilizing contiguous pointer mapping.
 */
void sum(double *c, double *b, int N) {
	int i;
	for (i = 0; i < N * N; i++) {
		*c += *b;
		c++;
		b++;
	}
}

/**
 * Block Logic: Transforms dot product sequences computing A^T * A incorporating upper triangular constraints.
 * Optimization: Replaces constant indexing with incremental caching across nested array traversals.
 */
void mulAtA(double *c, double *a, double *b, int N) {
	int i, j, k;
	for (j = 0; j < N; j++) {
		for (k = 0; k <= j; k++) {
			register int row_k = k * N;
			register double *pointer = &(c[j]);
			register double *elem_b = &(b[row_k + j]);
			register double *elem_a = &(a[row_k]);
			for (i = 0; i < N; i++) {
				*pointer += *elem_a * *elem_b;
				pointer += N;
				elem_a ++;
			}
		}
	}
}

/**
 * Block Logic: Evaluates BBt = B * B^T leveraging symmetric multiplication.
 * Optimization: Limits redundant index multiplication extracting static factors (like `row_i` and `row_j`) out of deeper loops.
 */
void mulBBt(double *c, double *a, double *b, int N) {
	int i, j, k;
	double sum;
	for (i = 0; i < N; i++) {
		register int row_i = i * N;
		for (j = 0; j < N; j++) {
			register int row_j = j * N;
			register double* elem_a = &(a[row_i]);
			register double* elem_b = &(b[row_j]);
			sum = 0;
			for (k = 0; k < N; k++) {
				sum += *elem_a * *elem_b;
				elem_a++;
				elem_b++;
			}
			*c = sum;
			c++;
		}
	}
}

/**
 * Block Logic: Derives A * (B * B^T) ensuring spatial cache awareness utilizing variables allocated within processor registers.
 */
void mulABBt(double *c, double *a, double *b, int N) {
	int i, j, k;
	for (i = 0; i < N; i++) {
		register int row_i = i * N;
		for (j = 0; j < N; j++) {
			double sum = 0;
			register double* elem_a = &(a[row_i + i]);
			register double* elem_b = &(b[row_i + j]);
			for (k = i; k < N; k++) {
				sum += *elem_a * *elem_b;
				elem_a++;
				elem_b += N;
			}
			*c = sum;
			c++;
		}
	}
}

/**
 * Functional Utility: Instantiates and maps dynamic memory regions calling underlying optimizations chronologically.
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
