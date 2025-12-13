
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	
	
	double *AB = malloc(N * N * sizeof(double));
	double *C = malloc(N * N * sizeof(double));
	int i, j, k, M;

	
	register double *pAB = AB;
	for (i = 0; i < N; ++i)
	{
		for (j = 0; j < N; ++j)
		{
			
			register double *pA = A + i * N + i; 
            register double *pB = B + j + i * N; 

			register double sum = 0.0;
			
			for (k = i; k < N; ++k)
			{
				sum += *pA * *pB;
				pA++;
				pB += N;
			}
			*pAB = sum;
			pAB++;
		}
	}

	
	register double *pC = C;
	for (i = 0; i < N; ++i)
	{
		for (j = 0; j < N; ++j)
		{
			
			register double *pAB = AB + i * N;
            register double *pB_t = B + j * N;

			register double sum = 0.0;

			for (k = 0; k < N; ++k)
			{
				sum += *pAB * *pB_t;
				pAB++;
				pB_t++;
			}
			
			*pC = sum;
			pC++;
		}
	}

	
    register double *pA_tA = C; 
    for (i = 0; i < N; ++i)
    {
    	for (j = 0; j < N; j++) 
		{
			
			register double *pA_t = A + i;
       		register double *pA = A + j;
		
			if (i >= j) {
                M = j;
            } else {
                M = i;
            }
		
			register double sum = 0.0;
			for (k = 0; k <= M; ++k)
        	{
				
           		sum += *pA_t * *pA;
				pA_t += N;
				pA += N;
        	}
        	*pA_tA += sum;
        	pA_tA++;
		}
	}                                                                                                                                                                                                                                                                                                                                                              
	
	free(AB);

	return C;
}
