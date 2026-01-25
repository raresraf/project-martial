
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	
	
	long nr_elem = N*N;
	register int i, j, k;
	double *C, *AB, *ABBt, *AtA, *At, *Bt;
	C = (double *)calloc(nr_elem, sizeof(double));
	AB = (double *)calloc(nr_elem, sizeof(double));
	ABBt = (double *)calloc(nr_elem, sizeof(double));
	AtA = (double *)calloc(nr_elem, sizeof(double));
	At = (double *)calloc(nr_elem, sizeof(double));
	Bt = (double *)calloc(nr_elem, sizeof(double));


	
	
	for (i = 0 ; i < N ; i ++)
	{
		register double *orig_pa = &A[i * N + i];
		for (j = 0 ; j < N ; j ++)
		{
			register double suma = 0;
			register double *pa = orig_pa;
            register double *pb = &B[i * N + j];
			for (k = i ; k < N ; k ++)
			{
				suma += *pa * *pb;
				pa++;
                pb += N;
			}
			AB[i * N + j] = suma;
		}
	}

	
	for (i = 0 ; i < N ; i ++)
	{
		for(j = 0 ; j < N ; j ++)
		{
			At[j * N + i] = A[i * N + j];
			Bt[j * N + i] = B[i * N + j];
		}
	}

	
	for (i = 0 ; i < N ; i ++)
	{
		register double *orig_pa = &AB[i * N];
		for (j = 0 ; j < N ; j ++)
		{
			register double suma = 0;
			register double *pa = orig_pa;
            register double *pb = &Bt[j];
			for (k = 0 ; k < N ; k ++)
			{
				
				suma += *pa * *pb;
				pa++;
                pb += N;
			}
			ABBt[i * N + j] = suma;
		}
	}

	


	
	for (i = 0 ; i < N ; i ++)
	{
		register double *orig_pa = &At[i * N];
		for (j = 0 ; j < N ; j ++)
		{
			register double suma = 0;
			register double *pa = orig_pa;
            register double *pb = &A[j];
			for (k = 0 ; k <= i ; k ++)
			{
				
				suma += *pa * *pb;
				pa++;
                pb += N;
			}
			AtA[i * N + j] = suma;
			C[i * N + j ] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	
	free(AB);
	free(ABBt);
	free(AtA);
	free(At);
	free(Bt);

	return C;

}
