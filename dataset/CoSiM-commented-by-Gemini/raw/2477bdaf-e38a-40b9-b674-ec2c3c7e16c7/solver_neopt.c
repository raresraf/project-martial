/**
 * @file solver_neopt.c
 * @brief A non-optimized, baseline implementation of a matrix solver.
 * @details This file provides a straightforward, unoptimized solution for the matrix equation
 * C = (A * B) * B' + A' * A, where A is an upper triangular matrix. The implementation
 * uses explicit loops and allocates several temporary matrices for intermediate results.
 */
#include "utils.h"

/**
 * @brief Solves C = (A * B) * B' + A' * A using a non-optimized, sequential approach.
 * @param N The dimension of the square matrices.
 * @param A A pointer to the input upper triangular matrix A (N x N).
 * @param B A pointer to the input matrix B (N x N).
 * @return A pointer to the resulting matrix C (N x N).
 *
 * @details The function performs the matrix operations in several distinct steps, using
 * temporary buffers to store intermediate results.
 * 1. Manually computes the transpose of A and B.
 * 2. Computes the intermediate product `result_1 = A * B`.
 * 3. Computes the first main term `result_2 = (A * B) * B'`.
 * 4. Computes the second main term `result_3 = A' * A`.
 * 5. Computes the final result `C = result_2 + result_3`.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER
");
	double* A_T = (double *)calloc(N * N , sizeof(double));
	double* B_T = (double *)calloc(N * N , sizeof(double));
	double* result_1 = (double *)calloc(N * N , sizeof(double));
	double* result_2 = (double *)calloc(N * N , sizeof(double));
	double* result_3 = (double *)calloc(N * N , sizeof(double));
	double* C = (double *)calloc(N * N , sizeof(double));
	int i ,j,k;

	
	/**
	 * Block Logic: Step 1: Manually compute transposes of A and B.
	 * Time Complexity: O(N^2)
	 */
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			A_T[N * i + j] = A[N * j + i];
			B_T[N * i + j] = B[N * j + i];
		}
	}
	
	
	/**
	 * Block Logic: Step 2: Compute the intermediate product result_1 = A * B.
	 * The inner loop `k` starts from `i`, which is an optimization for
	 * multiplication with an upper triangular matrix A.
	 * Time Complexity: O(N^3), with a lower constant factor.
	 */
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			for(k = i; k < N; k++)
			{
				result_1[N * i + j] += A[N * i + k] * B[N * k + j];
			}
		}
	}
	
	
	/**
	 * Block Logic: Step 3: Compute the first term result_2 = (A * B) * B'.
	 * This multiplies the intermediate result `result_1` with `B_T`.
	 * Time Complexity: O(N^3)
	 */
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			for(k = 0; k < N; k++)
			{
				result_2[N * i + j] += result_1[N * i + k] * B_T[N * k + j];
			}
		}
	}
	
	
	/**
	 * Block Logic: Step 4: Compute the second term result_3 = A' * A.
	 * This multiplies `A_T` with `A`. The loop `k <= (i < j ? i : j)` is
	 * an optimization equivalent to `k <= min(i,j)`, which correctly
	 * handles the multiplication of a lower triangular (A') and an upper
	 * triangular (A) matrix.
	 * Time Complexity: O(N^3), with a lower constant factor.
	 */
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			for(k = 0; k <= (i < j ? i : j); k++)
			{
				result_3[N * i + j] += A_T[N * i + k] * A[N * k + j];
			}
		}
	}


	
	/**
	 * Block Logic: Step 5: Compute the final result C = result_2 + result_3.
	 * This is an element-wise addition of the two main terms.
	 * Time Complexity: O(N^2)
	 */
	for(i = 0 ; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			C[N * i + j] = result_2[N * i + j] + result_3[N * i + j];
		}
	}	


	// Free all dynamically allocated temporary matrices.
	free(A_T);
	free(B_T);
	free(result_1);
	free(result_2);
	free(result_3);
	return C;
}
