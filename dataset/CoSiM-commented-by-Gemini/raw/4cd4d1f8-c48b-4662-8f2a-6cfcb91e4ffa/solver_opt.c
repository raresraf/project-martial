
#include "utils.h"


double* my_solver(int N, double *A, double* B)
{
	double *AtA, *C, *ABBt, *AB;

	
	C = malloc(N * N * sizeof(*C));
	if (NULL == C)
		exit(1);

	AtA = malloc(N * N * sizeof(*AtA));
	if (NULL == AtA)
		exit(1);

	AB = malloc(N * N * sizeof(*AB));
	if (NULL == AB)
		exit(1);

	ABBt = malloc(N * N * sizeof(*ABBt));
	if (NULL == ABBt)
		exit(1);
	
	for (register int i = 0; i < N; i++)
		for (register int j = 0; j < N; j++){
			AB[i * N + j] = 0;
			ABBt[i * N + j] = 0;
			AtA[i * N + j] = 0;
			C[i * N + j] = 0;
		}

	
	for (register int i = 0; i < N; i++) {
		register double *orig_pa = &A[i * N + i];
		for (register int j = 0; j < N; j++){
			register double suma = 0.0;
			register double *pa = orig_pa;
    		register double *pb = &B[i * N + j];
			for (int k = i; k < N; k++){
				suma += *pa* *pb;
				pa++;
				pb +=N;
			}
			AB[i * N + j] = suma;
		}
	}
	
	for (register int i = 0; i < N; i++) {
		register double *orig_pa = &AB[i * N];
		for (register int j = 0; j < N; j++) {
			register double suma = 0.0;
			register double *pa = orig_pa;
    		register double *pb = &B[j * N];
			for (register int k = 0; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb+=1;
			}
			ABBt[i * N + j] = suma;
		}
	}

	
	for (register int i = 0; i < N; i++) {
		register double *orig_pa = &A[i];
		for (register int j = 0; j < N; j++){
			register double suma = 0.0;
			register double *pa = orig_pa;
    		register double *pb = &A[j];
			for (register int k = 0; k <= j; k++){
				suma += *pa* *pb;
				pa+=N;
				pb+=N;
			}
			AtA[i * N + j] = suma;
		}
	}
	
	
	for (register int i = 0; i < N; i++) {
		register double *pa = &ABBt[i * N];
		register double *pb = &AtA[i * N];
		for (register int j = 0; j < N; j++){
			C[i * N + j] = *pa + *pb;
			pa++;
			pb++;
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);
	return C;
}
