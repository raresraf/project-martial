/**
 * @file solver_opt.c
 * @brief Manually optimized matrix solver implementation.
 * Features register blocking, loop reordering, and cache-friendly data access patterns.
 */

#include "utils.h"


/**
 * @brief Computes C = At * A + A * B * Bt.
 * Allocates memory dynamically and executes matrix operations.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	double *C;
	double *B_t;
	double *A_t;
	double *AB;

	register int size = N * N * sizeof(*C);

	C = malloc(size);
	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if (NULL == C)
		exit(EXIT_FAILURE);

	B_t = malloc(size);
	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if (NULL == B_t)
		exit(EXIT_FAILURE);
	
	A_t = malloc(size);
	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if (NULL == A_t)
		exit(EXIT_FAILURE);

	AB = malloc(size);
	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if (NULL == AB)
		exit(EXIT_FAILURE);

	
	/* Pre-condition: Arrays allocated. Invariant: transposing and copying matrices A and B */
	register int i;
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i != N; ++i) {
		register double *B_t_ptr = B_t + i;  
		register double *A_t_ptr = A_t + i;  

		register double *B_ptr = B + i * N;  
		register double *A_ptr = A + i * N;  
		
		register int j;
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j != N; ++j, A_t_ptr += N, ++A_ptr, B_t_ptr += N, ++B_ptr) {
			*B_t_ptr = *B_ptr;
			*A_t_ptr = *A_ptr;
		}
	}

	
	
	register double *A_ptr = A;
	register double *B_ptr = B;
	register double *B_copy = B;  
	register double *A_copy = A;  
	register double *AB_ptr;
	register int k;
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (k = 0; k != N; ++k, B_copy += N, ++A_copy)
	{
		A_ptr = A_copy;
		register double *AB_copy = AB;
		register int i;

		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (i = 0; i != N; ++i, A_ptr += N, AB_copy += N)
		{
			AB_ptr = AB_copy;
			B_ptr = B_copy;

			register int j;

			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (j = 0; j != N; ++j, ++B_ptr, ++AB_ptr)
			{
				*AB_ptr += *A_ptr * *B_ptr;
			}
		}
	}

	
	register double *C_ptr;
	register double *AB_copy = AB;  
	AB_ptr = AB;
	B_ptr = B_t;
	B_copy = B_t;  
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (k = 0; k != N; ++k, B_copy += N, ++AB_copy)
	{
		AB_ptr = AB_copy;
		register double *C_copy = C;

		register int i;

		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (i = 0; i != N; ++i, AB_ptr += N, C_copy += N)
		{
			C_ptr = C_copy;  
			B_ptr = B_copy;

			register int j;

			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (j = 0; j != N; ++j, ++B_ptr, ++C_ptr)
			{
				*C_ptr += *AB_ptr * *B_ptr;
			}
		}
	}

	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i != N; ++i) {
		register double *C_ptr = C + i * N;
		register double *A_tA_copy = A_t + i * N;
		register int j;

		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j != N; ++j, ++C_ptr) {
			
			register double result = 0;

			register double *A_tA_ptr = A_tA_copy;

			 
			register double *A_ptr = A_t + j * N;

			register int k;

			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (k = 0; k != N; ++k, ++A_tA_ptr, ++A_ptr) {
				result += *A_tA_ptr * *A_ptr;
			}

			*C_ptr += result;
		}
	}

	free(B_t);
	free(A_t);
	free(AB);
	
	return C;	
}
A_tA_ptr * *A_ptr;
			}

			*C_ptr += result;
		}
	}

	free(B_t);
	free(A_t);
	free(AB);
	
	return C;	
}
