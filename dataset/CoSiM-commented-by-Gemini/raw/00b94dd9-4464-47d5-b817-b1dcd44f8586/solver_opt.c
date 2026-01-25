/**
 * @file solver_opt.c
 * @brief A manually "optimized" implementation of a matrix solver.
 *
 * This file provides a version of the solver that attempts to optimize
 * the matrix operations through manual pointer arithmetic and the use of
 * the `register` keyword. It aims to compute the same expression as the
 * non-optimized solver.
 */
#include "utils.h"
#include <string.h>


/**
 * @brief Computes the transpose of a square matrix.
 * @param M The input square matrix of size N*N.
 * @param N The dimension of the matrix.
 * @return A new matrix which is the transpose of M. The caller is responsible for freeing this memory.
 */
static double *get_transpose(double *M, int N)
{
	double *tr = calloc(N * N, sizeof(double));
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			tr[i * N + j] = M[j * N + i];
		}
	}
	return tr;
}


/**
 * @brief Solves a matrix equation of the form: (A*B)*B^T + A^T*A using manually optimized loops.
 * @param N The dimension of the square matrices A and B.
 * @param A Input matrix A, assumed to be upper triangular.
 * @param B Input matrix B.
 * @return A new matrix containing the result of the computation. The caller is responsible for freeing this memory.
 *
 * @b Algorithm: This function follows the same computational steps as the non-optimized version
 * but uses pointer arithmetic and the `register` keyword in an attempt to improve performance.
 * Note: The implementation of the first multiplication (`A*B`) appears to be buggy due to incorrect
 * pointer logic, but the intended calculation is preserved in this documentation.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	double *first_mul = calloc(N * N, sizeof(double));
	double *first_mul_aux = calloc(N * N, sizeof(double));
	double *second_mul = calloc(N * N, sizeof(double));
	double *Bt = get_transpose(B, N);
	double *At = get_transpose(A, N);

	
	/**
	 * @brief This block is intended to compute `first_mul = A * B`.
	 * It leverages A being upper triangular. However, the pointer arithmetic seems flawed
	 * and does not correctly implement matrix multiplication.
	 */
	for (int i = 0; i < N; ++i) {
		register double *aux = &A[i * N];
		for (int j = 0; j < N; ++j) {
			register double *collumn = &B[j];
			register double *line = aux;

			register double rez = 0;
			for (int k = i; k < N; ++k, line++, collumn += N) {
				rez += *(line + i) * *(collumn + i * N);
			}
			first_mul[i * N + j] = rez;
		}
	}


	
	/**
	 * @brief This block computes `first_mul_aux = first_mul * B^T`.
	 * This calculates `(A * B) * B^T`, although `first_mul` may be incorrect from the previous step.
	 */
	for (int i = 0; i < N; ++i) {
		register double *aux = &first_mul[i * N];
		for (int j = 0; j < N; ++j) {
			register double *line = aux;
			register double *collumn = &Bt[j];
			register double res = 0;

			for (int k = 0; k < N; ++k, line++, collumn += N) {
				res += *line * *collumn;
			}
			first_mul_aux[i * N + j] = res;
		}
	}

		
	/**
	 * @brief This block computes `second_mul = A^T * A`.
	 * It leverages A being upper triangular (so A^T is lower triangular).
	 */
	for (int i = 0; i < N; ++i) {
		register double *aux = &At[i * N];
		for (int j = 0; j < N; ++j) {
			register double *line = aux;
			register double *collumn = &A[j];
			register double res = 0;

			for (int k = 0; k <= i; ++k, line++, collumn += N) {
				res += *line * *collumn;
			}
			second_mul[i * N + j] = res;
		}
	}

	
	// Final summation: `first_mul_aux = first_mul_aux + second_mul`
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			first_mul_aux[i * N + j] += second_mul[i * N + j];
		}
	}

	
	free(first_mul);
	free(second_mul);
	free(At);
	free(Bt);

	return first_mul_aux;	
}