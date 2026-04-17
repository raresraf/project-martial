/**
 * @file solver_opt.c
 * @brief A micro-optimized implementation of a matrix solver that contains logical errors.
 * @details This file attempts to solve the matrix equation C = (A * B) * B' + A' * A
 * using manual optimization techniques like the 'register' keyword and pointer arithmetic.
 * However, the core logic for matrix multiplication is flawed.
 */
#include "utils.h"
#include <string.h>


/**
 * @brief Attempts to solve C = (A * B) * B' + A' * A using a flawed, micro-optimized approach.
 * @param N The dimension of the square matrices.
 * @param A A pointer to the input upper triangular matrix A (N x N).
 * @param B A pointer to the input matrix B (N x N).
 * @return A pointer to the resulting matrix C (N x N).
 *
 * @warning The implementation of the matrix multiplication steps in this function is incorrect.
 * Instead of performing dot products, the loops compute element-wise products of rows and
 * sum the results, which does not correspond to standard matrix multiplication. The comments
 * below describe the intended operation versus the actual (buggy) implementation.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER
");
	register size_t i, j, k;

	
	double *C = calloc(sizeof(double), N * N);
	double *transA = calloc(sizeof(double), N * N);
	double *transB = calloc(sizeof(double), N * N);

	/**
	 * Block Logic: Step 1: Manually compute transposes of A and B using pointer arithmetic.
	 * This part correctly transposes the matrices.
	 */
	for (i = 0; i < N; ++i) {
		register double *ptransA = transA + i;
		register double *ptransB = transB + i;
		register double *pA = A + N * i;
		register double *pB = B + N * i;
		for (j = 0; j < N; ++j) {
			*ptransA = *pA;
			*ptransB = *pB;
			ptransA += N;
			ptransB += N;
			pA++;
			pB++;
		}
	}

	
	/**
	 * Block Logic: Step 2: Intended to compute C = A * B.
	 * @bug This loop is incorrect. It calculates `C(i,j) = sum_{k=i..N-1} A(i,k) * B'(j,k)`.
	 * This is not a matrix product. It's a sum of an element-wise product between a
	 * partial row of A and a partial row of B' (a partial column of B).
	 */
	for (i = 0; i < N; ++i) {
		register double *pc = C + N * i;
		register double *orig_pa = A + N * i;
		for (j = 0; j < N; ++j) {
			register double suma = 0.0;
			register double *pa = orig_pa;
			register double *pb = transB + N * j;
			for (k = i; k < N; ++k) {
				suma += *(pa + k) * *(pb + k);
			}
			*pc = suma;
			pc++;
		}
	}

	double *tmp = calloc(sizeof(double), N * N);
	memcpy(tmp, C, N * N * sizeof(double));

	
	/**
	 * Block Logic: Step 3: Intended to compute C = (A * B) * B'.
	 * @bug This loop is also incorrect. It calculates `C(i,j) = sum_{k=0..N-1} tmp(i,k) * B(j,k)`.
	 * This is not a matrix product with a transpose. It is a sum of an element-wise product
	 * between a row of the previous result (`tmp`) and a row of `B`.
	 */
	for (i = 0; i < N; ++i) {
		register double *pc = C + N * i;
		register double *orig_tmp = tmp + N * i;
		for (j = 0; j < N; ++j) {
			register double suma = 0;
			register double *ptmp = orig_tmp;
			register double *pb = B + N * j;
			for (k = 0; k < N; ++k) {
				suma += *(ptmp + k) * *(pb + k);
			}
			*pc = suma;
			pc++;
		}
	}

	memcpy(tmp, C, N * N * sizeof(double));
	

	double *tmp2 = calloc(sizeof(double), N * N);
	
	/**
	 * Block Logic: Step 4: Intended to compute tmp2 = A' * A.
	 * @bug This loop is also incorrect. It calculates `tmp2(i,j) = sum_{k=0..N-1} A'(i,k) * A'(j,k)`.
	 * This is the dot product of row `i` of A' with row `j` of A', not A' * A.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_transA = transA + N * i;
		register double *ptmp2 = tmp2 + N * i;
		for (j = 0; j < N; ++j) {
			register double suma = 0;
			register double *ptransA1 = orig_transA;
			register double *ptransA2 = transA + N * j;
			for (k = 0; k < N; ++k) {
				suma += *(ptransA1 + k) * *(ptransA2 + k);
			}
			*ptmp2 = suma;
			ptmp2++;
		}
	}
	
	/**
	 * Block Logic: Step 5: Final element-wise addition.
	 * This correctly adds the results of the previous flawed calculations.
	 */
	for (i = 0; i < N; ++i) {
		register double *pc = C + N * i;
		register double *ptmp = tmp + N * i;
		register double *ptmp2 = tmp2 + N * i;
		for (j = 0; j < N; ++j) {
			*pc = *ptmp + *ptmp2;
			pc++, ptmp++, ptmp2++;
		}
	}

	free(tmp);
	free(tmp2);
	free(transA);
	free(transB);

	return C;
}
