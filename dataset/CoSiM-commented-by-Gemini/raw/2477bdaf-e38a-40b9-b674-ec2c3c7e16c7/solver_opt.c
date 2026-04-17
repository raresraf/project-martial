/**
 * @file solver_opt.c
 * @brief A micro-optimized implementation of a matrix solver.
 * @details This file provides a solution for the matrix equation C = (A * B) * B' + A' * A,
 * where A is an upper triangular matrix. The implementation uses manual optimization
 * techniques such as the 'register' keyword and pointer arithmetic to improve performance
 * over the baseline version.
 */
#include "utils.h"

/**
 * @brief Solves C = (A * B) * B' + A' * A using a micro-optimized, sequential approach.
 * @param N The dimension of the square matrices.
 * @param A A pointer to the input upper triangular matrix A (N x N).
 * @param B A pointer to the input matrix B (N x N).
 * @return A pointer to the resulting matrix C (N x N).
 *
 * @details This function is an optimized version of the non-optimized solver. It uses
 * `register` hints for frequently used variables and direct pointer manipulation to
 * iterate through matrices, aiming to reduce addressing overhead.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER
");
	double* A_T = (double *)calloc(N * N , sizeof(double));
	double* B_T = (double *)calloc(N * N , sizeof(double));
	double* result_1 = (double *)calloc(N * N , sizeof(double));
	double* result_2 = (double *)calloc(N * N , sizeof(double));
	double* result_3 = (double *)calloc(N * N , sizeof(double));
	double* C = (double *)calloc(N * N , sizeof(double));
	register int i ,j,k;
	register double buff = 0.0;
	register double* orig_pa;
	register double* pa;
	register double* pb;
	register double* pa_t;
	register double* pb_t;
	
	
	/**
	 * Block Logic: Step 1: Manually compute transposes of A and B using pointer arithmetic.
	 * This loop iterates through columns of the source matrices and writes to rows of the
	 * destination matrices.
	 * Time Complexity: O(N^2)
	 */
	for(i = 0; i < N; i++)
	{ 
		pa = &A[N * 0 + i];
		pb = &B[N * 0 + i];
		pa_t = &A_T[N * i + 0];
		pb_t = &B_T[N * i + 0];
		for(j = 0; j < N; j++)
		{
			* pa_t = * pa;
			* pb_t = * pb;
			pa = pa + N; 
			pb = pb + N; 
			pa_t++;
			pb_t++; 
		}
	}
	
	
	/**
	 * Block Logic: Step 2: Compute the intermediate product result_1 = A * B.
	 * This optimized loop calculates the dot product of a row of A with a column of B.
	 * `pa` iterates through the row of A, while `pb` iterates through the column of B.
	 * The loop for `k` starts from `i`, exploiting the upper-triangular nature of A.
	 * Time Complexity: O(N^3), with a reduced operation count.
	 */
	for(i = 0; i < N; i++)
	{
		orig_pa = &A[N * i + i];
		for(j = 0; j < N; j++)
		{
			pa = orig_pa;
			pb = &B[N * i + j];
			buff = 0.0;
			for(k = i; k < N; k++)
			{
				buff += *pa * *pb;
				pa++;
				pb += N;
			}
			result_1[N * i + j] = buff;
		}
	}
	
	
	/**
	 * Block Logic: Step 3: Compute the first term result_2 = (A * B) * B'.
	 * This multiplies the intermediate result `result_1` with `B_T`. `pa` iterates
	 * through a row of `result_1`, and `pb` iterates through a column of `B_T`.
	 * Time Complexity: O(N^3)
	 */
	for(i = 0; i < N; i++)
	{
		orig_pa = &result_1[N *i + 0];
		for(j = 0; j < N; j++)
		{
			pa = orig_pa;
			pb = &B_T[N * 0 + j];
			buff = 0.0;
			for(k = 0; k < N; k++)
			{
				buff += *pa * *pb;
				pa++;
				pb += N;
			}
			result_2[N * i + j] = buff;
		}
	}
	
	
	/**
	 * Block Logic: Step 4: Compute the second term result_3 = A' * A.
	 * This multiplies `A_T` with `A`. `pa` iterates a row of `A_T` (a column of A)
	 * and `pb` iterates a column of `A`. The `k` loop correctly reflects the
	 * sparsity of the operation.
	 * Time Complexity: O(N^3), with a reduced operation count.
	 */
	for(i = 0; i < N; i++)
	{
		orig_pa = &A_T[N * i + 0];
		for(j = 0; j < N; j++)
		{
			pa = orig_pa;
			pb = &A[N * 0 + j];	
			buff = 0.0;
			for(k = 0; k <= (i < j ? i : j); k++)
			{
				buff += *pa * *pb;
				pa++;
				pb += N;
			}
			result_3[N * i + j] = buff;
		}
	}

	
	/**
	 * Block Logic: Step 5: Compute the final result C = result_2 + result_3.
	 * Element-wise addition using pointer arithmetic.
	 * Time Complexity: O(N^2)
	 */
	for(i = 0 ; i < N; i++)
	{
		pa = &result_2[N * i + 0];
		pb = &result_3[N * i + 0];
		for(j = 0; j < N; j++)
		{
			C[N * i + j] = *pa + *pb;
			pa++;
			pb++;
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
