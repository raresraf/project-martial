/**
 * @file solver_neopt.c
 * @brief Unoptimized standard implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Naive multi-loop dense matrix multiplication with upper triangular conditions.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store the intermediate matrices.
 */

#include "utils.h"





void add(double* result, double* A, double* B, int N) {
    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for (int i = 0; i < N; i++) {
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for(int j = 0; j < N; j++) {
            result[i * N + j] = A[i * N + j] + B[i * N + j];
		}
    }
    
};

void prodTriSub(double* result, double* A, double* B, int N) {
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (int i = 0; i < N; i++) {
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for (int j = 0; j < N; j++) {
            double aux = 0;
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for (int k = 0; k < N; k++){
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k >= i) {
                	aux = aux + A[i * N + k] * B[k * N + j];
				}
            }
            result[i * N + j] = aux;
        }
    }
}


void prodTriLow(double* result, double* A, double* B, int N) {
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (int i = 0; i < N; i++) {
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for (int j = 0; j < N; j++) {
            double aux = 0;
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for (int k = 0; k < N; k++){
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k <= i) {
                	aux = aux + A[i * N + k] * B[k * N + j];
				}
            }
            result[i * N + j] = aux;
        }
    }
}



void prod(double* result, double* A, double* B, int N) {
    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for (int i = 0; i < N; i++) {
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for (int j = 0; j < N; j++) {
            double aux = 0;
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for (int k = 0; k < N; k++){
                aux = aux + A[i * N + k] * B[k * N + j];
            }
            result[i * N + j] = aux;
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
	double* C = (double*) malloc(N * N * sizeof(double));
	double* At = (double*) malloc(N * N * sizeof(double));
	double* Bt = (double*) malloc(N * N * sizeof(double));

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(int i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for(int j = 0; j < N; j++) {
			At[i * N + j] = A[i + N * j];
			Bt[i * N + j] = B[i + N * j];
		}
	}


	

	double* X = (double*) malloc(N * N * sizeof(double));
	prodTriSub(X, A, B, N);

	
	double* Y = (double*) malloc(N * N * sizeof(double));
	prod(Y, X, Bt, N);

	
	double* Z = (double*) malloc(N * N * sizeof(double));
	prodTriLow(Z, At, A, N);


	
	add(C, Y, Z, N);

	
	free(At);
	free(Bt);
	free(X);
	free(Z);
	free(Y);
	return C;
}
