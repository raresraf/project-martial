
#include "utils.h"
#include <stdlib.h>



int min(int a, int b) {
    return a < b ? a : b;
}

void allocate_matrices(int N, double **C, double **AB, double **ABBt,
	double **AtA, double **A_t, double **B_t)
{
	*C = malloc(N * N * sizeof(**C));
	if (NULL == *C)
		exit(EXIT_FAILURE);

	*AB = calloc(N * N, sizeof(**AB));
	if (NULL == *AB)
		exit(EXIT_FAILURE);

	*ABBt = calloc(N * N, sizeof(**ABBt));
	if (NULL == *ABBt)
		exit(EXIT_FAILURE);

	*AtA = calloc(N * N, sizeof(**AtA));
	if (NULL == *AtA)
		exit(EXIT_FAILURE);

	*A_t = calloc(N * N, sizeof(**AtA));
	if (NULL == *AtA)
		exit(EXIT_FAILURE);

	*B_t = calloc(N * N, sizeof(**B_t));
	if (NULL == *B_t)
		exit(EXIT_FAILURE);
}

double* my_solver(int N, double *A, double* B) {
	double *C;
	double *AB;
	double *ABBt;
	double *AtA;

	double *A_t;
	double *B_t;

	allocate_matrices(N, &C, &AB, &ABBt, &AtA, &A_t, &B_t);

	
	register int i;
	for (i = 0; i < N; ++i) {
		register double *A_t_ptr = A_t + i;  
		register double *B_t_ptr = B_t + i;  

		register double *A_ptr = A + i * N;  
		register double *B_ptr = B + i * N;  

		register int j;
		for (j = 0; j < N; ++j, A_t_ptr += N,
			B_t_ptr += N, ++A_ptr, ++B_ptr) {
			*A_t_ptr = *A_ptr;
			*B_t_ptr = *B_ptr;
		}
	}
	
	for (i = 0; i < N; ++i) {
		register double *AtA_ptr = AtA + i * N;  
		
		register int j;
		for (j = 0; j < N; ++j, ++AtA_ptr) {
			
			register double result = 0;

			
			register double *A_ptr = A + j % N;
		        register double *A_t_ptr = A_t + i * N;  
			
			register int k;
			for (k = 0; k < min(i + 1, j + 1); ++k, 
                        A_t_ptr = A_t + i * N + k, A_ptr = A + k * N + j) {
				result += *A_t_ptr * *A_ptr;
        		}
			*AtA_ptr = result;
		}
	}
	
    
	for (i = 0; i < N; ++i) {
		register double *AB_ptr = AB + i * N;  
		
		register int j;	
		for (j = 0; j < N; ++j, ++AB_ptr) {
			
			register double result = 0;

			
			register double *A_ptr = A + i * N + i;
	                register double *B_ptr = B + j % N + i * N;  
			
			register int k;
			for (k = i; k < N; ++k, ++A_ptr, B_ptr += N) {
				result += *A_ptr * *B_ptr;
            		}
			*AB_ptr = result;
		}
	}

    
	for (i = 0; i < N; ++i) {
		register double *C_ptr = C + i * N;  
		
		register int j;
		for (j = 0; j < N; ++j, ++C_ptr) {
			
			register double result = 0;

			
			register double *AB_ptr = AB + i * N;
		        register double *B_t_ptr = B_t + j % N;  
			
			register int k;
			for (k = 0; k < N; ++k, ++AB_ptr, B_t_ptr += N) {
				result += *AB_ptr * *B_t_ptr;
            		}
			*C_ptr = result;
		}
	}

    
	for (i = 0; i < N; ++i) {
		register double *C_ptr = C + i * N;  
		register double *AtA_ptr = AtA + i * N;  
		
		register int j;
        	for(j = 0; j < N; ++j, C_ptr++, AtA_ptr++) {
            		*C_ptr += *AtA_ptr;
        	}
	}
	
	free(AB);
	free(ABBt);
	free(AtA);

	free(A_t);
	free(B_t);
	return C;	
}
