/**
 * @file solver_opt.c
 * @brief Cache-optimized matrix solver implementation.
 * Integrates loop reordering, pointer arithmetic, and register variables for cache-friendly data access.
 */
#include "utils.h"


/**
 * @brief Executes the optimized matrix equation solver.
 * @param N The dimension of the square matrices A and B.
 * @param A Pointer to the first input matrix.
 * @param B Pointer to the second input matrix.
 * @return A dynamically allocated array containing the result matrix D.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	int i, j, k;
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
     * @pre Matrices A and B are initialized. Loop layout optimizes row-major caching.
     * @post C stores the upper-triangular dot product of A and B using precomputed row pointers.
     */
    for (i = 0; i < N; i++) {
		/// Registers pointer offsets directly avoiding repetitive row address calculation.
		register double *p_c = C + i * N;
		register double *p_a = A + i * N;
		for (k = i; k < N; k++) {
			register double *p_b = B + k * N;
            for (j = 0; j < N; j++) {
				/// Resolves 2D indices via pointer arithmetic to ensure continuous block access.
				*(p_c + j) += *(p_a + k) * *(p_b + j);
            }
        }
    }
	
    /**
     * @pre Intermediate product C is ready.
     * @post D accumulates inner product iterations leveraging a temporary accumulation register.
     */
    for (i = 0; i < N; i++) {
		register double *p_c = C + i * N;
		register double *p_d = D + i * N;
        for (j = 0; j < N; j++) {
			/// Loop-invariant accumulator mapped to hardware registers for spatial locality optimization.
			register double rez = 0.0;
			register double *p_b = B + j * N;
            for (k = 0; k < N; k++) {
				rez += *(p_c + k) * *(p_b + k);
            }
			*(p_d + j) = rez;
        }
    }
	
    /**
     * @pre Matrix D is populated with initial transformations.
     * @post Updates D with transpose-equivalent inner combinations.
     */
	for (k = 0; k < N; k++) {
		register double *p_a = A + k * N;
		for (i = k; i < N; i++) {
			register double *p_d = D + i * N;
            for (j = k; j < N; j++) {
				/// Caches offset dereferences utilizing previously evaluated pointer offsets.
				*(p_d + j) += *(p_a + i) * *(p_a + j);
            }
		}
	}
	free(C);
	return D;
}
