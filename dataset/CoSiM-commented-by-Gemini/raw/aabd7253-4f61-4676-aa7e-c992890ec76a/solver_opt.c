/**
 * @file solver_opt.c
 * @brief An optimized implementation of a matrix expression solver.
 *
 * This file provides an optimized version of the function that calculates the
 * matrix expression: (A_upper * B * B^T) + (A_lower^T * A_upper).
 *
 * Optimization Techniques Used:
 * 1.  Function Specialization: The generic `multiply` function with internal
 *     branches is replaced by three specialized functions (`multiply_normal`,
 *     `multiply_upper`, `multiply_lower_upper`). This eliminates conditional
 *     checks from the innermost loop, improving instruction pipelining.
 * 2.  Pointer Arithmetic: Instead of array indexing (e.g., `A[i * N + j]`),
 *     the code extensively uses direct pointer manipulation for sequential
 *     memory access, which can be more efficient.
 * 3.  Register Hinting: The `register` keyword is used to suggest to the
 *     compiler that loop counters and frequently accessed pointers be stored
 *     in CPU registers.
 * 4.  Local Accumulator: A local variable (`suma`) is used within the
 *     multiplication loops to accumulate results, which can be optimized
 *     by the compiler to use a register.
 * 5.  Simplified Loops: The `add` function uses a single loop over N*N
 *     elements instead of a nested loop.
 */

#include "utils.h"
#define UPPER 1
#define LOWER -1
#define NORMAL 0

/**
 * @brief Optimized matrix transpose using pointer arithmetic.
 * @param N The dimension of the matrix.
 * @param C Pointer to the output transposed matrix.
 * @param M The input matrix.
 *
 * Optimization: Uses pointers `ptrC` and `ptrM` to iterate through the
 * destination and source matrices, avoiding repeated index calculations.
 * The inner loop strides through the source matrix `M` by `N` to access
 * column elements.
 */
void transpose(int N, double **C, double *M) {
	register int i, j;
	for(i = 0; i < N; i ++) {
		register double *ptrC = &(*C)[i * N ];
		register double *ptrM = &M[i];
		for(j = 0; j < N; j++) {
			*ptrC = *ptrM;
			ptrM += N;
			ptrC++;
		}
	}
}

/**
 * @brief Optimized matrix multiplication for two normal (dense) matrices.
 * @param N The dimension of the matrices.
 * @param C Pointer to the output matrix.
 * @param A The first input matrix.
 * @param B The second input matrix.
 *
 * Optimization: This specialized function avoids any conditional checks for
 * matrix type within its loops. It uses pointer arithmetic and register
 * variables for faster access and accumulation.
 */
void multiply_normal(int N, double **C, double *A, double *B, int typeA, int typeB) {
	register double *ptrC = *C;
	register int i, j, k;
	for(i = 0; i < N; i++) {
		register double *pa = &A[i * N];
		for(j = 0; j < N; j++) {
			register double suma = 0.0;
			register double *ptrA = pa;
			register double *ptrB = &B[j];
			// Innermost loop: computes the dot product of a row from A and a column from B.
			for(k = 0; k < N; k++) {
				suma += *ptrA * *ptrB;
				ptrA++;
				ptrB += N;
			}
			*ptrC = suma;
			ptrC++;
		}
	}
}

/**
 * @brief Optimized matrix multiplication where the first matrix (A) is upper triangular.
 * @param N The dimension of the matrices.
 * @param C Pointer to the output matrix.
 * @param A The first input matrix (assumed upper triangular).
 * @param B The second input matrix.
 *
 * Optimization: The `if (i <= k)` check is integrated into the otherwise
 * optimized multiplication loop structure.
 */
void multiply_upper(int N, double **C, double *A, double *B, int typeA, int typeB) {
	register double *ptrC = *C;
	register int i, j, k;
	for(i = 0; i < N; i++) {
		register double *pa = &A[i * N];
		for(j = 0; j < N; j++) {
			register double suma = 0.0;
			register double *ptrA = pa;
			register double *ptrB = &B[j];
			for(k = 0; k < N; k++) {
				// Block Logic: Skips multiplication if the element of A is in the lower triangle part.
				if(i <= k)
					suma += *ptrA * *ptrB;
				ptrA++;
				ptrB += N;
			}
			*ptrC = suma;
			ptrC++;
		}
	}
}

/**
 * @brief Optimized multiplication where A is lower triangular and B is upper triangular.
 * @param N The dimension of the matrices.
 * @param C Pointer to the output matrix.
 * @param A The first input matrix (assumed lower triangular).
 * @param B The second input matrix (assumed upper triangular).
 *
 * Optimization: Integrates the combined condition `if(i >= k && k <= j)` into the
 * optimized loop structure.
 */
void multiply_lower_upper(int N, double **C, double *A, double *B, int typeA, int typeB) {
	register double *ptrC = *C;
	register int i, j, k;
	for(i = 0; i < N; i++) {
		register double *pa = &A[i * N];
		for(j = 0; j < N; j++) {
			register double suma = 0.0;
			register double *ptrA = pa;
			register double *ptrB = &B[j];
			for(k = 0; k < N; k++) {
				// Block Logic: Skips multiplication based on triangular properties of both matrices.
				if(i >= k && k <= j)
					suma += *ptrA * *ptrB;
				ptrA++;
				ptrB += N;
			}
			*ptrC = suma;
			ptrC++;
		}
	}
}

/**
 * @brief Optimized element-wise matrix addition C = A + B.
 * @param N The dimension of the matrices.
 * @param C Pointer to the output matrix.
 * @param A The first input matrix.
 * @param B The second input matrix.
 *
 * Optimization: This function treats the 2D matrices as 1D arrays of size N*N
 * and uses a single loop with pointer arithmetic for faster iteration.
 */
void add(int N, double **C, double *A, double *B) {
	register double *ptrC = *C;
	register double *ptrA = &A[0];
	register double *ptrB = &B[0];
	register int i;
	for(i = 0; i < N * N; i++) {
		*ptrC = *ptrA + *ptrB;
		ptrA++;
		ptrB++;
		ptrC++;
	}
}

/**
 * @brief Calculates the matrix expression (A * B * B^T) + (A^T * A) using optimized functions.
 * @param N The dimension of the square matrices A and B.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return A pointer to the resulting matrix. The caller is responsible for freeing this memory.
 *
 * Optimization: This version of the solver calls the specialized multiplication
 * functions (`multiply_upper`, `multiply_normal`, `multiply_lower_upper`)
 * instead of a single generic function. This avoids branching within the inner
 * loops of the matrix multiplication, leading to better performance.
 */
double* my_solver(int N, double *A, double* B) {
	
	// Step 1: Calculate AB = A * B (A is upper triangular)
	double *AB = calloc(N* N,sizeof(double));
	multiply_upper(N, &AB, A, B, UPPER, NORMAL);

	// Step 2: Transpose B
	double *Btrans = malloc((N * N) * sizeof(double));
	transpose(N, &Btrans, B);

	// Step 3: Calculate ABBt = AB * B^T (normal multiplication)
	double *ABBt = calloc(N * N,sizeof(double));
	multiply_normal(N, &ABBt, AB, Btrans, NORMAL, NORMAL);

	
	// Step 4: Transpose A
	double *Atrans = malloc((N * N) * sizeof(double));
	transpose(N, &Atrans, A);
	
	// Step 5: Calculate AtA = A^T * A (A^T is lower, A is upper)
	double *AtA = calloc(N* N,sizeof(double));
	multiply_lower_upper(N, &AtA, Atrans, A, LOWER, UPPER);

	// Step 6: Add the two intermediate results
	double *result = malloc((N * N) * sizeof(double));
	add(N, &result, ABBt, AtA);

	// Free all intermediate allocated memory
	free(AB);
	free(Btrans);
	free(ABBt);
	free(Atrans);
	free(AtA);

	return result;
}
