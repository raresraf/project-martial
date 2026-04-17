/**
 * @file solver_neopt.c
 * @brief Unoptimized, baseline matrix solver implementation.
 * Uses naive O(N^3) triple nested loop iterations without caching optimizations or SIMD directives.
 */
#include "utils.h"


/**
 * @brief Executes the unoptimized matrix equation solver.
 * @param N The dimension of the square matrices A and B.
 * @param A Pointer to the first input matrix.
 * @param B Pointer to the second input matrix.
 * @return A dynamically allocated array containing the result matrix D.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	int k, j, i;
	double *C = (double*) calloc(N * N, sizeof(double));
    if (C == NULL) {
		printf("Calloc failed!\n");
      	exit(1);
	}
    double *D = (double*) calloc(N * N, sizeof(double));
    if (D == NULL) {
		printf("Calloc failed!\n");
      	exit(1);
	}

    /**
     * @pre Matrices A and B are populated. C is zero-initialized.
     * @post Matrix C contains the upper triangular product C = A * B.
     */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = i; k < N; k++) {
                /// Non-obvious array offset calculation mapping 2D indices to a 1D linear array buffer.
                *(C + i * N + j) += *(A + i * N + k) * *(B + k * N + j);
            }
        }
    }
    
    /**
     * @pre Matrix C contains intermediate result. Matrix D is zero-initialized.
     * @post Matrix D aggregates the product D = C * B^T or similar matrix combination.
     */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                /// Standard 2D to 1D mapping with unoptimized strided memory access pattern.
                *(D + i * N + j) += *(C + i * N + k) * *(B + j * N + k);
            }
        }
    }
    
    /**
     * @pre Matrix D contains intermediate results.
     * @post Matrix D adds the product A^T * A.
     */
	for (k = 0; k < N; k++) {
		for (i = k; i < N; i++) {
            for (j = k; j < N; j++) {
                /// Pointer arithmetic accessing matrix elements with column-major-like traversal on A.
                *(D + i * N + j) += *(A + k * N + i) * *(A + k * N + j);
            }
		}
	}
	free(C);
	return D;
}
