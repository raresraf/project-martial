/**
 * @file solver_neopt.c
 * @brief Naive, non-optimized implementation of a matrix equation solver.
 * @details This file provides a basic, loop-based implementation for computing
 * the matrix equation: result = (A * B) * B^T + A^T * A. It assumes that the
 * input matrix A is upper triangular. This version is intended for correctness
 * testing and as a baseline for performance comparison against optimized versions.
 * The implementation uses three nested loops for matrix multiplication, leading to
 * a time complexity of O(N^3).
 */
#include "utils.h"
#include <string.h>

/**
 * @brief Computes the matrix expression (A * B) * B^T + A^T * A using naive loops.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N, row-major, upper triangular).
 * @param B A pointer to the second input matrix (N x N, row-major).
 * @return A pointer to the resulting N x N matrix. The caller is responsible for freeing this memory.
 */
double* my_solver(int N, double *A, double* B) {
	// Allocate and zero-initialize memory for intermediate and final results.
	double *ab = (double *) calloc(N * N, sizeof(double));
	double *abbt = (double *) calloc(N * N, sizeof(double));
	double *ata = (double *) calloc(N * N, sizeof(double));
	double *result =  (double *) calloc(N * N, sizeof(double));

	
	/**
	 * Block Logic: Compute ab = A * B.
	 * Assumes A is an upper triangular matrix, so the inner loop starts from k = i.
	 * Time Complexity: O(N^3)
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = i; k < N; ++k) {
				ab[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	/**
	 * Block Logic: Compute abbt = ab * B^T.
	 * The access pattern `B[j * N + k]` corresponds to accessing B transposed.
	 * Time Complexity: O(N^3)
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				abbt[i * N + j] += ab[i * N + k] * B[j * N + k];
			}
		}
	}
	
	
	/**
	 * Block Logic: Compute ata = A^T * A.
	 * The access pattern `A[k * N + i]` corresponds to accessing A transposed.
	 * Assumes A is upper triangular, so k <= i is sufficient.
	 * Time Complexity: O(N^3)
	 */
	for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                        for (int k = 0; k <= i; ++k) {
                                ata[i * N + j] += A[k * N + i] * A[k * N + j];
                        }
                }
        }
	
	
	/**
	 * Block Logic: Sum the intermediate results.
	 * result = abbt + ata => (A * B) * B^T + A^T * A
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			result[i * N + j] = abbt[i * N + j] + ata[i * N + j];
		}
	}
	
	// Free the memory allocated for intermediate matrices.
	free(ab);
	free(abbt);
	free(ata);
	return result;
}
