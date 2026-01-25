
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	
	
	double *AB = malloc(N * N * sizeof(double));
	double *C = malloc(N * N * sizeof(double));
	int i, j, k, M;
		
	
	for (i = 0; i < N; ++i)
	{
		for (j = 0; j < N; ++j)
		{
			AB[i * N + j] = 0.0;
			for (k = i; k < N; ++k)
			{
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; ++i)
	{
		for (j = 0; j < N; ++j)
		{
			C[i * N + j] = 0.0;
			for (k = 0; k < N; ++k)
			{
				C[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for (i = 0; i < N; ++i)
        {
                for (j = 0; j < N; ++j)
                {
                        if (i >= j) {
                                M = j;
                        } else {
                                M = i;
                        }

                        for (k = 0; k <= M; ++k)
                        {
                                C[i * N + j] += A[k * N + i] * A[k * N + j];
                        }
                }
        }

	
	free(AB);

	return C;
}
