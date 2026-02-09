/**
 * @file solver_opt.c
 * @brief Semantic documentation for solver_opt.c.
 *        This is a placeholder. Detailed semantic analysis will be applied later.
 */

#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	double *C = (double *)malloc(N * N * sizeof(double));
    double *D = (double *)malloc(N * N * sizeof(double));
    register int i, j, k;

    
    /**
     * @brief [Functional Utility for for]: Describe purpose here.
     */
    for (i = 0; i < N; i++) {
    	register double *orig_pb1 = &B[i * N];
        /**
         * @brief [Functional Utility for for]: Describe purpose here.
         */
        for (j = i; j < N; j++) {
            register double aux = 0;
            register double *pb1 = orig_pb1;
            register double *pb2 = &B[j * N];

            /**
             * @brief [Functional Utility for for]: Describe purpose here.
             */
            for (k = 0; k < N; k++) {
                aux += *pb1 * *pb2;
                pb1++;
                pb2++;
            }

            C[i * N + j] = aux;
            // Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            // Invariant: State condition that holds true before and after each iteration/execution
            if (i != j)
                C[j * N + i] = aux;

        }
    }

    
    /**
     * @brief [Functional Utility for for]: Describe purpose here.
     */
    for (i = 0; i < N; i++) {
    	register double *orig_pa = &A[i * (N + 1)];
        /**
         * @brief [Functional Utility for for]: Describe purpose here.
         */
        for (j = 0; j < N; j++) {
            register double aux = 0;
            register double *pa = orig_pa;
            register double *pc = &C[j * N + i];

            /**
             * @brief [Functional Utility for for]: Describe purpose here.
             */
            for (k = i; k < N; k++) {
                aux += *pa * *pc;
                pa++;
                pc++;
            }

            D[i * N + j] = aux;
        }
    }

    
    /**
     * @brief [Functional Utility for for]: Describe purpose here.
     */
    for (i = 0; i < N; i++) {
    	register double *orig_pa = &A[i];
        /**
         * @brief [Functional Utility for for]: Describe purpose here.
         */
        for (j = i; j < N; j++) {
            register double aux = 0;
            register double *pa1 = orig_pa;
            register double *pa2 = &A[j];

            /**
             * @brief [Functional Utility for for]: Describe purpose here.
             */
            for (k = 0; k <= i; k++) {
                aux += *pa1 * *pa2;
                pa1 += N;
                pa2 += N;
            }

            /**
             * @brief [Functional Utility for if]: Describe purpose here.
             */
            if (i != j) {
                D[j * N + i] += aux;
            }

            D[i * N + j] += aux;
        }
    }

    free(C);
	return D;	
}
