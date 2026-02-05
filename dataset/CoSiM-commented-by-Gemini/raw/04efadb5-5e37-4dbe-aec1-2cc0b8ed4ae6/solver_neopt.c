
/**

 * @file solver_neopt.c

 * @brief A non-optimized, straightforward implementation of a matrix equation solver.

 *

 * This file contains a basic implementation for solving the matrix equation

 * C = A*B*B' + A'*A. It uses nested loops for matrix multiplication and

 * does not employ any performance optimization techniques. This version is

 * likely intended as a baseline for correctness verification against more

 * optimized implementations.

 */

#include "utils.h"





/**

 * @brief Solves the matrix equation C = A*B*B' + A'*A using basic loops.

 *

 * This function calculates the result of the matrix equation by breaking it down

 * into intermediate steps:

 * 1. AB = A * B (where A is assumed to be upper triangular)

 * 2. ABBt = AB * B'

 * 3. AtA = A' * A (where A is assumed to be upper triangular)

 * 4. C = ABBt + AtA

 * Each step is computed using explicit, non-optimized for-loops.

 *

 * @param N The dimension of the square matrices A and B.

 * @param A A pointer to the first input matrix (N x N), treated as upper triangular.

 * @param B A pointer to the second input matrix (N x N).

 * @return A pointer to the resulting matrix C, or NULL if memory allocation fails.

 */

double * my_solver(int N, double * A, double * B) {

    int i, j, k;



    double * C;

    double * AB;

    double * ABBt;

    double * AtA;



    // Functional Utility: Allocates memory for the final result matrix C and intermediate matrices.

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

    

    // Block Logic: Computes the product of A and B, storing it in AB.

    // Assumes A is an upper triangular matrix, hence k starts from i.

    for (i = 0; i < N; i++)

        for (j = 0; j < N; j++)

            for (k = i; k < N; k++)

                AB[i * N + j] += A[i * N + k] * B[k * N + j];



    

    // Block Logic: Computes the product of AB and the transpose of B, storing it in ABBt.

    // The B[j * N + k] access pattern effectively uses B'.

    for (i = 0; i < N; i++)

        for (j = 0; j < N; j++)

            for (k = 0; k < N; k++)

                ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];



    

    // Block Logic: Computes the product of the transpose of A and A, storing it in AtA.

    // The A[k * N + i] access pattern effectively uses A'.

    // Assumes A is upper triangular, so the inner loop could be optimized.

    for (i = 0; i < N; i++)

        for (j = 0; j < N; j++)

            for (k = 0; k <= j; k++)

                AtA[i * N + j] += A[k * N + i] * A[k * N + j];



    

    // Block Logic: Computes the final matrix C by summing the intermediate results.

    for (i = 0; i < N; i++)

        for (j = 0; j < N; j++)

            C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];



    // Functional Utility: Frees memory allocated for intermediate matrices.

    free(AB);

    free(ABBt);

    free(AtA);



    return C;

}
