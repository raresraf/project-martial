/**
 * @file solver_neopt.c
 * @brief A non-optimized C implementation of a matrix solver.
 *
 * This file provides a `my_solver` implementation that computes the expression
 * `C = (A * B) * B^T + A^T * A` using a single function with several sequential
 * blocks of nested loops for the matrix operations.
 */
#include "utils.h"


/**
 * @brief Computes a matrix expression using basic, non-optimized C loops.
 *
 * This function calculates the result of `C = (A * B) * B^T + A^T * A` for given
 * N x N matrices A and B. It allocates several temporary matrices and performs
 * all calculations using explicit, triply-nested loops.
 *
 * @param N The dimension of the square matrices.
 * @param A A pointer to the N x N input matrix A.
 * @param B A pointer to the N x N input matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the final result.
 *         The caller is responsible for freeing this memory.
 *
 * @note The computation is broken down into three main loop blocks:
 * 1. Computes `AtA = A^T * A`.
 * 2. Computes `AB = A * B`, assuming A is an upper triangular matrix.
 * 3. Computes `ABBt = AB * B^T` and immediately adds the `AtA` result to produce
 *    the final result matrix `res`. This combined final step improves data locality.
 */
double* my_solver(int N, double *A, double* B) {
	
	printf("NEOPT SOLVER
");
	double *AtA = calloc(N * N, sizeof(double));
	double *AB = calloc(N * N, sizeof(double));
	double *ABBt = calloc(N * N, sizeof(double));
	double *res = calloc(N * N, sizeof(double));

	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				if (k > j) {
					break;
				}
				AtA[i*N + j] += A[k*N + i] * A[k*N + j];
			}
		}
	}

	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				if (k < i) {
					continue;
				}
				AB[i*N + j] += A[i*N + k] * B[k*N + j];
			}
		}
	}

	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				ABBt[i*N + j] += AB[i*N + k] * B[j*N + k];
			}
			res[i*N + j] = ABBt[i*N + j] + AtA[i*N + j];
		}
	}

	free(AtA);
	free(AB);
	free(ABBt);

	return res;
}
