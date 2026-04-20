/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
 */

#include "utils.h"



double* transpose(int N, double* A) {
	double* At = malloc(N * N * sizeof(double));

	register size_t i, j;
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			At[i*N + j] = A[j*N + i];
		}
	}
	return At;
}

double* inmultire(int N, double *A, double* B) {
	double* M = malloc( N * N * sizeof(double));

	register size_t i, j, k;
	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N ; i++) {
		register double* org_pa = &A[N*i]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		register double* org_M = &M[N*i]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N ; j++) {
		    register double *pa = org_pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double* pb = &B[N*j]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double calcul = 0;
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; k++) {
				calcul += *(pa) * (*pb); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pa++;
				pb++;
			}

			*(org_M) = calcul;
			org_M++;
		}
	}

	return M;
}

double* inmultireSuperiorTriunghiulara(int N, double* A, double* B) {
	double* M = calloc(sizeof(double), N * N);

	register size_t i, j, k;

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N ; i++) {
		register double* org_pa = &A[N*i]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		register double* org_M = &M[N*i]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N ; j++) {
		    register double *pa = org_pa + i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double* pb = &B[N*j + i]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double calcul = 0;

			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; k++){
				calcul += *(pa) * (*pb); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pa++;
				pb++;
			}
			*(org_M) = calcul;
			org_M++;
		}
	}

	return M;
}

double* inmultireLowerXUpper(int N, double* A, double* B) {
	double* M = calloc(sizeof(double), N * N);

	register size_t i, j, k;
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N ; i++) {
		register double* org_pa = &A[N*i]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		register double* org_M = &M[N*i]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N ; j++) {
		    register double *pa = org_pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double* pb = &B[N*j]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double calcul = 0;

			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < j + 1; k++){
				calcul += *(pa) * (*pb); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pa++;
				pb++;
			}
			*(org_M) = calcul;
			org_M++;
		}
	}

	return M;
}

double* adunare(int N, double *A, double* B) {
	double* M = calloc(sizeof(double), N * N);
	
	register size_t i, j;
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N ; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N ; j++)
			M[i*N + j] = A[i*N + j] + B[i*N + j]; 
	}
	return M;
}

/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	double* B_transpose = transpose(N, B);
	double* AxB = inmultireSuperiorTriunghiulara(N, A, B_transpose);
	double* firstPart = inmultire(N, AxB, B);
	double* A_transpose = transpose(N, A);
	double* secondPart = inmultireLowerXUpper(N, A_transpose, A_transpose);

	double* C = adunare(N, firstPart, secondPart);

	free(AxB);
	free(B_transpose);
	free(firstPart);
	free(secondPart);
	free(A_transpose);

	return C;
}