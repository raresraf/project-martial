
#include "utils.h"



double* my_solver(int N, double *A, double* B) {
	
	double *AB;
	double *ABB_t;
	double *A_tA;
	double *Res;
	double *A_t;
	double *B_t;
	int i, j, k;
	double suma;

	AB = malloc(N * N * sizeof(*AB));
	if(NULL == AB)
		exit(EXIT_FAILURE);

	ABB_t = malloc(N * N * sizeof(*ABB_t));	
	if(NULL == ABB_t)
		exit(EXIT_FAILURE);

	A_tA = malloc(N * N * sizeof(*A_tA));
	if(NULL == A_tA)
		exit(EXIT_FAILURE);

	A_t = malloc(N * N * sizeof(*A_t));
	if(NULL == A_t)
		exit(EXIT_FAILURE);

	B_t = malloc(N * N * sizeof(*B_t));
	if(NULL == B_t)
		exit(EXIT_FAILURE);

	Res = malloc(N * N * sizeof(*Res));
	if(NULL == Res)
		exit(EXIT_FAILURE);

	for(i = 0; i < N; i++)
		for(j = 0; j < N; j++)
			B_t[i * N + j] = B[j * N + i];

	

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++){
			suma = 0;
			for (k = i; k < N; k++)
				suma += A[i * N + k] * B[k * N + j];
			AB[i * N + j] = suma;
		}

	

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++){
			suma = 0;
			for (k = 0; k < N; k++)
				suma += AB[i * N + k] * B_t[k * N + j];
			ABB_t[i * N + j] = suma;
		}

	

	for(i = 0; i < N; i++)
		for(j = 0; j < N; j++)
			A_t[i * N + j] = A[j * N + i];

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++){
			suma = 0;
			for (k = 0; k <= j; k++)
				suma += A_t[i * N + k] * A[k * N + j];
			A_tA[i * N + j] = suma;
		}

	

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			Res[i * N + j] = ABB_t[i * N + j] + A_tA[i * N + j];


	free(AB);
	free(A_t);
	free(B_t);
	free(A_tA);
	free(ABB_t);

	return Res;
}
