/**
 * @raw/8890c616-fbd7-4a4e-98ec-ad5ad4a09f55/solver_opt.c
 * @brief Optimized dense matrix multiplication solver computing C = (A * B) * B^T + A^T * A.
 * Algorithm: Focuses on cache-locality and explicit loop optimizations via transposition, pointers, and CPU registers.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ tracking internal partial product evaluations and localized matrix caches.
 */
#include <stdlib.h>
#include "utils.h"

/**
 * Functional Utility: Initializes dynamically allocated matrix buffers and caches for intermediate outputs.
 */
void allocate(int N, double **C, double **AB, double **ABB_t,
			 double **A_tA, double **A_t, double **B_t)
{
	*C = calloc(N * N, sizeof(double));
	if (NULL == *C)
		exit(EXIT_FAILURE);

	*AB = calloc(N * N, sizeof(double));
	if (NULL == *AB)
		exit(EXIT_FAILURE);

	*ABB_t = calloc(N * N, sizeof(double));
	if (NULL == *ABB_t)
		exit(EXIT_FAILURE);

	*A_tA = calloc(N * N, sizeof(double));
	if (NULL == *A_tA)
		exit(EXIT_FAILURE);

	*A_t = malloc(N * N * sizeof(double));
	if (NULL == *A_t)
		exit(EXIT_FAILURE);

	*B_t = malloc(N * N * sizeof(double));
	if (NULL == *B_t)
		exit(EXIT_FAILURE);
}

double* my_solver(register int N, register double *A, 
				 register double* B)
{
	double *C, *AB, *ABB_t, *A_tA, *A_t, *B_t;

	register int i, j, k;

	allocate(N, &C, &AB, &ABB_t, &A_tA, &A_t, &B_t);

	/**
	 * Block Logic: Computes AB = A * B.
	 * Optimization: Pointer arithmetic avoids integer multiplication bounds overhead. Local sum stored in scalar `res`.
	 */
	for (i = 0; i < N; ++i) {
		register double *ptrA = A + i * N;

		for (j = 0; j < N; ++j) {
			register double *pA = ptrA + i;
			register double *pB = B + i * N + j;
			register double res = 0;

			for (k = i; k < N; ++k) {
				res += *pA * *pB;
				pA++;
				pB += N;
			}

			AB[i * N + j] = res;
		}
	}

	/**
	 * Block Logic: Calculates cache-friendly memory arrangements for computing consecutive matrix multiplications.
	 * Invariant: Transposes matrix A and matrix B sequentially.
	 */
	for (i = 0; i != N; ++i) {
		register double *ptr_A_t = A_t + i;
		register double *ptr_B_t = B_t + i;

		register double *ptr_A = A + i * N;
		register double *ptr_B = B + i * N;

		for (j = 0; j != N; ++j, ptr_A_t += N,
			ptr_B_t += N, ++ptr_A, ++ptr_B) {
			*ptr_A_t = *ptr_A;
			*ptr_B_t = *ptr_B;
		}
	}

	/**
	 * Block Logic: Evaluates concurrent accumulations of A^T * A and (A * B) * B^T directly into the target C matrix.
	 * Optimization: Transposed vectors enable sequential data retrieval across consecutive memory addresses.
	 */
	for (i = 0; i < N; ++i) {
		register double *ptrA_t = A_t + i * N;
		register double *ptrAB = AB + i * N;
		register double *ptr_res = C + i * N;

		for (j = 0; j < N; ++j) {
			register double *pA_t = ptrA_t;
			register double *A_ptr = A + j;
			register double *pAB = ptrAB;
			register double *pB_t = B_t + j;

			register double res = 0;
			register double result = 0;

			for (k = 0; k < N; ++k) {
				res += *pAB * *pB_t;
				pAB++;
				pB_t += N;
				result += *pA_t * *A_ptr;
				pA_t++;
				A_ptr += N;
			}

			*ptr_res = result + res;
			ptr_res++;
		}
	}

	free(AB);
	free(ABB_t);
	free(A_tA);
	free(A_t);
	free(B_t);

	return C;
}
