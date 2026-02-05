
/**

 * @file solver_opt.c

 * @brief An optimized implementation of a matrix equation solver.

 *

 * This file provides a solver for the matrix equation C = A*B*B' + A'*A.

 * It employs micro-optimizations such as using register-based loop counters and

 * local accumulator variables within the loops to potentially improve performance

 * by suggesting to the compiler which variables should be prioritized for

 * register allocation, thus reducing memory access overhead.

 */

#include "utils.h"





/**

 * @brief Solves the matrix equation C = A*B*B' + A'*A with micro-optimizations.

 *

 * This function calculates the result of the matrix equation by breaking it down

 * into four main steps, each performed in a separate loop structure:

 * 1. AB = A * B (assuming A is upper triangular)

 * 2. ABBt = AB * B'

 * 3. AtA = A' * A (assuming A is upper triangular)

 * 4. C = ABBt + AtA

 *

 * It uses local accumulator variables within the loops to minimize writes to

 * the result matrix until the inner loop is complete.

 *

 * @param N The dimension of the square matrices A and B.

 * @param A A pointer to the first input matrix (N x N), treated as upper triangular.

 * @param B A pointer to the second input matrix (N x N).

 * @return A pointer to the resulting matrix C, or NULL if memory allocation fails.

 */

double * my_solver(int N, double * A, double * B) {



    double * C;

    double * AB;

    double * ABBt;

    double * AtA;

    // Optimization: Suggests loop counters be stored in registers for faster access.

    register int i = 0;

    register int j = 0;

    register int k = 0;



    // Functional Utility: Allocates memory for the final result and intermediate matrices.

    C = calloc(N * N, sizeof( * C)); 

    if (C == NULL) {

        return NULL;

    }

    AB = calloc(N * N, sizeof( * C)); 

    if (AB == NULL) {

        return NULL;

    }

    ABBt = calloc(N * N, sizeof( * C)); 

    if (ABBt == NULL) {

        return NULL;

    }

    AtA = calloc(N * N, sizeof( * C)); 

    if (AtA == NULL) {

        return NULL;

    }



    

    // Block Logic: Computes the product of A and B (AB = A * B).

    // A is treated as an upper triangular matrix (k starts from i).

    // A local accumulator `sumAB` is used to reduce memory writes inside the loop.

    for (i = 0; i < N; i++) {

        for (j = 0; j < N; j++) {

            register double sumAB = 0.0;

            for (k = i; k < N; k++)

                sumAB += A[i * N + k] * B[k * N + j];

            AB[i * N + j] = sumAB;

        }

    }

    

    // Block Logic: Computes the product of AB and the transpose of B (ABBt = AB * B').

    // The access pattern B[j * N + k] effectively uses B'.

    // An accumulator `sumABBt` is used for performance.

    for (i = 0; i < N; i++) {

        for (j = 0; j < N; j++) {

            register double sumABBt = 0.0;

            for (k = 0; k < N; k++)

                sumABBt += AB[i * N + k] * B[j * N + k];

            ABBt[i * N + j] = sumABBt;

        }

    }



    

    // Block Logic: Computes the product of the transpose of A and A (AtA = A' * A).

    // The access pattern A[k * N + i] effectively uses A'.

    // An accumulator `sumAtA` is used for performance.

    for (i = 0; i < N; i++) {

        for (j = 0; j < N; j++) {

            register double sumAtA = 0.0;

            for (k = 0; k <= j; k++)

                sumAtA += A[k * N + i] * A[k * N + j];

            AtA[i * N + j] = sumAtA;

        }

    }



    

    // Block Logic: Computes the final matrix C by summing the intermediate results.

    for (i = 0; i < N; i++)

        for (j = 0; j < N; j++)

            C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];



    // Functional Utility: Frees memory for intermediate matrices.

    free(AB);

    free(ABBt);

    free(AtA);



    return C;

}
