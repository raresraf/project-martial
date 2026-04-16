/**
 * @file solver_neopt.c
 * @brief A non-optimized, naive implementation of a matrix solver.
 *
 * This file provides a straightforward C implementation of the `my_solver` function.
 * It uses simple nested loops for all matrix operations, serving as a baseline
 * reference for correctness and a point of comparison against optimized versions
 * (like the BLAS or other optimized implementations). It calculates the expression:
 * C = (A * B) * B^T + A^T * A.
 */
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIN(a,b) (((a)<(b))?(a):(b))


/**
 * @brief Computes a matrix expression using basic, non-optimized C loops.
 *
 * This function calculates the result of `C = (A * B) * B^T + A^T * A` for given
 * N x N matrices A and B. It allocates several temporary matrices to hold
 * intermediate results and performs all calculations using explicit, triply-nested
 * loops without any performance optimizations like blocking or vectorization.
 *
 * @param N The dimension of the square matrices.
 * @param A A pointer to the N x N input matrix A. Assumed to be upper triangular.
 * @param B A pointer to the N x N input matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the final result.
 *         The caller is responsible for freeing this memory.
 *
 * @note The computation is broken down into four main steps:
 * 1. `prod11 = A * B` (Matrix multiplication, exploiting the upper triangular nature of A).
 * 2. `prod12 = prod11 * B^T` (Multiplication with the transpose of B).
 * 3. `prod2 = A^T * A` (Multiplication of A-transpose with A).
 * 4. `rez = prod12 + prod2` (Element-wise addition to get the final result).
 */
double* my_solver(int N, double *A, double* B) {
	double *rez;

	double *prod11, *prod12;
	double *prod2;

	int i, j, k;

	rez = calloc(N * N, sizeof(double));
	if (rez == NULL)
		exit(EXIT_FAILURE);

	prod11 = calloc(N * N, sizeof(double));
	if (prod11 == NULL)
		exit(EXIT_FAILURE);

	prod12 = calloc(N * N, sizeof(double));
	if (prod12 == NULL)
		exit(EXIT_FAILURE);

	prod2 = calloc(N * N, sizeof(double));
	if (prod2 == NULL)
		exit(EXIT_FAILURE);

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				prod11[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				prod12[i * N + j] += prod11[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k <= MIN(i, j); k++) {
				prod2[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			rez[i * N + j] = prod12[i * N + j] + prod2[i * N + j];
		}
	}

	free(prod11);
	free(prod12);
	free(prod2);
	return rez;
}
