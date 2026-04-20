/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
 */

#include "utils.h"




void mult(int N, double *mat1, double *mat2, double *rez) {
	int i, j, k;

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		register int i_x_N = i * N;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (k = 0; k < N; k++) {
			register double ct = mat1[i_x_N + k];
			register int k_x_N = k * N;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (j = 0; j < N; j++) {
				rez[i_x_N + j] += ct * mat2[j + k_x_N];
			}
		}
	}
}


void mult_triang_inf_triang_sup(int N, double *mat1, double *mat2, double *rez) {
	int i, j, k;

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		register int i_x_N = i * N;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			register double suma = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k <= (j < i ? j : i); k++) {
				suma += mat1[i_x_N + k] * mat2[k * N + j] ;
			}
			rez[i_x_N + j] = suma;
		}
	}
}


void mult_triang_sup_norm(int N, double *mat1, double *mat2, double *rez) {
	int i, j, k;

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		register int i_x_N = i * N;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (k = i; k < N; k++) {
			register double ct = mat1[i_x_N + k];
			register int k_x_N = k * N;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (j = 0; j < N; j++) {
				rez[i_x_N + j] += ct * mat2[j + k_x_N];
			}
		}
	}
}


void add(int N, double *mat1, double *mat2) {
	int i, j;

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		register int i_x_N = i * N;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			mat1[i_x_N + j] += mat2[i_x_N + j];
		}
	}
}


void transp(int N, double *mat, int triang, double *rez) {
	int i, j;

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register int i_x_N = i * N;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0 + i * triang; j < N; ++j) {
			rez[i + N * j] = mat[i_x_N + j];
		}
	}
}

/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	double *rez, *A_t, *B_t, *aux1, *aux2;

	B_t = (double*) calloc(N * N, sizeof(double));
	A_t = (double*) calloc(N * N, sizeof(double));
	aux1 = (double*) calloc(N * N, sizeof(double));
	aux2 = (double*) calloc(N * N, sizeof(double));
	rez = (double*) calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (!B_t || !A_t || !aux1 || !aux2 || !rez)
		return NULL;

	
	transp(N, B, 0, B_t);

	
	transp(N, A, 1, A_t);

	
	mult_triang_sup_norm(N, A, B, aux1); 
	
	
	mult(N, aux1, B_t, rez);

	
	mult_triang_inf_triang_sup(N, A_t, A, aux2);

	
	add(N, rez, aux2);

	free(B_t);
	free(A_t);
	free(aux1);
	free(aux2);
	
	return rez;
}

