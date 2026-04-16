/**
 * @file solver_neopt.c
 * @brief A modular, non-optimized C implementation of a matrix solver.
 *
 * This file provides a `my_solver` implementation that breaks down the matrix
 * computation `C = (A * B) * B^T + A^T * A` into a series of distinct helper
 * functions. Each helper function performs a basic matrix operation (e.g.,
 * multiplication, addition) using simple, non-optimized nested loops.
 *
 * This modular approach makes the code easier to read than a single monolithic
 * function but does not employ any performance optimization techniques.
 */
#include "utils.h"



int min(int i, int j){
    return (i < j) ? i : j;
}


/**
 * @brief Computes the matrix product C = A^T * A, where A is upper triangular.
 *
 * @param N The dimension of the matrices.
 * @param A A pointer to the N x N upper triangular input matrix A.
 * @param B A pointer to the N x N input matrix B (should be the same as A).
 * @return A pointer to a newly allocated N x N matrix containing the result.
 */
double* mult_At_A(int N, double* A, double* B) {
    int i, j, k;
    double *c;
    c = calloc(N * N, sizeof(double));
    
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            for (k = 0; k < min(i + 1, j + 1); k++){
                c[i * N + j] += A[k * N + i] * B[k * N + j];
            }
        }
    }
    return c;
}


/**
 * @brief Computes the matrix product C = A * B^T.
 *
 * @param N The dimension of the matrices.
 * @param A A pointer to the N x N input matrix A.
 * @param B A pointer to the N x N input matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the result.
 */
double* mult_mult1_Bt(int N, double* A, double* B) {
    int i, j, k;
    double *c;
    c = calloc(N * N, sizeof(double));
    
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            for (k = 0; k < N; k++){
                c[i * N + j] += A[i * N + k] * B[j * N + k];
            }
        }
    }
    return c;
}


/**
 * @brief Computes the element-wise sum of two matrices, C = A + B.
 *
 * @param N The dimension of the matrices.
 * @param A A pointer to the N x N input matrix A.
 * @param B A pointer to the N x N input matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the sum.
 */
double* add_matrix(int N, double* A, double* B) {
    int i, j;
    double *c;
    c = calloc(N * N, sizeof(double));
    
    for (i = 0; i < N; i++){
        for(j = 0; j < N; j++) 
            c[i * N + j] += A[i * N + j] + B[i * N + j];
    }
    return c;
}


/**
 * @brief Computes the matrix product C = A * B, where A is upper triangular.
 *
 * @param N The dimension of the matrices.
 * @param A A pointer to the N x N upper triangular input matrix A.
 * @param B A pointer to the N x N input matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the result.
 */
double* mult_A_B(int N, double* A, double* B) {
    int i, j, k;
    double *c;
    c = calloc(N * N, sizeof(double));
    
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            for (k = i; k < N; k++){
                c[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
    return c;
}

/**
 * @brief Allocates memory for several matrices.
 * @note This function appears to be unused in this file.
 */
void allocate_matrices(int N, double **C, double **AB, double **ABBt,
	double **AtA)
{
	*C = malloc(N * N * sizeof(**C));
	if (NULL == *C)
		exit(EXIT_FAILURE);

	*AB = calloc(N * N, sizeof(**AB));
	if (NULL == *AB)
		exit(EXIT_FAILURE);

	*ABBt = calloc(N * N, sizeof(**ABBt));
	if (NULL == *ABBt)
		exit(EXIT_FAILURE);

	*AtA = calloc(N * N, sizeof(**AtA));
	if (NULL == *AtA)
		exit(EXIT_FAILURE);
}

/**
 * @brief Computes the expression C = (A * B) * B^T + A^T * A by calling modular helper functions.
 *
 * This function orchestrates the matrix computation by calling a sequence of helper
 * functions, each responsible for one part of the overall expression.
 *
 * @param N The dimension of the square matrices.
 * @param A A pointer to the N x N input matrix A.
 * @param B A pointer to the N x N input matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the final result.
 *         The caller is responsible for freeing this memory.
 *
 * @note The sequence of operations is:
 * 1. `AB = A * B`
 * 2. `ABBt = AB * B^T`
 * 3. `AtA = A^T * A`
 * 4. `C = ABBt + AtA`
 */
double* my_solver(int N, double *A, double *B) {
	double *AB;
    double *ABBt;
    double *AtA;
    double *C;

    AB = mult_A_B(N, A, B);
    ABBt = mult_mult1_Bt(N, AB, B);
    AtA = mult_At_A(N, A, A);
    C = add_matrix(N, ABBt, AtA);

    free(AB);
    free(ABBt);
    free(AtA);

    return C;
}
