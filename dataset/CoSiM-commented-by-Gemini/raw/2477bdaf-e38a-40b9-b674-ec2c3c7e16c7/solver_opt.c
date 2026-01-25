
#include "utils.h"



double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
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


	
	free(A_T);
	free(B_T);
	free(result_1);
	free(result_2);
	free(result_3);
	return C;
}
