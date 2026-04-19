/**
 * @raw/8a7b5be2-e828-4ed1-add7-9ff564200fc3/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver computing C = (A * B) * B^T + A^T * A.
 * Algorithm: Standard nested loops traversing the matrices without structural assumptions beyond the inner bounds.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ for intermediate accumulation matrices.
 */

#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	/**
	 * Functional Utility: Allocates memory for final result C and intermediate matrices AB, first, second.
	 */
	double *AB;
	double *first;
	double *second;
	double *C;
	AB = calloc(N * N, sizeof(double));
	if(AB == NULL){
		exit(-1);
	}
	first = calloc(N * N, sizeof(double));
	if(first == NULL){
		exit(-1);
	}
	second = calloc(N * N, sizeof(double));
	if(second == NULL){
		exit(-1);
	}
	C = calloc(N * N, sizeof(double));
	if(C == NULL){
		exit(-1);
	}
	int i, j, k = 0;
    
	/**
	 * Block Logic: Computes intermediate matrix product AB = A * B.
	 * Invariant: Exploits the upper triangular nature of matrix A by initializing k = i.
	 */
	for(i = 0; i < N; i++){
		for(j = 0; j < N; j++){
			for(k = i; k < N; k++){
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
	
	/**
	 * Block Logic: Computes first = AB * B^T.
	 * Invariant: Accesses matrix B in transposed layout through [j * N + k] index calculation.
	 */
	for(i = 0; i < N; i++){
		for(j = 0; j < N; j++){
			for(k = 0; k < N; k++){
				first[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}
    
	/**
	 * Block Logic: Computes second = A^T * A.
	 * Invariant: Adjusts loop bounds to respect the upper triangular structure (`k <= i`).
	 */
	for(i = 0; i < N; i++){
		for(j = 0; j < N; j++){
			for(k = 0; k <= i; k++){
				second[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}
    
	/**
	 * Block Logic: Final summation step synthesizing C = first + second.
	 * Invariant: Element-wise addition combining partial evaluations.
	 */
    for(i = 0; i < N; i++){
		for(j = 0; j < N; j++){
			C[i * N + j] = first[i * N + j] + second[i * N + j];
		}
	}

	free(AB);
	free(first);
	free(second);
	return C;
}
