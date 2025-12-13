
#include "utils.h"



double* my_solver(int N, double *A, double* B) {
	
	double *AB;
	double *ABB_t;
	double *A_tA;
	double *A_t;
	double *B_t;
	double *Res;

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

	for(register int i = 0; i < N; i++)
		for(register int j = 0; j < N; j++){
			B_t[i * N + j] = B[j * N + i];
			A_t[i * N + j] = A[j * N + i];
		}

	

	for (register int i = 0; i < N; i++){
		register double *orig_pa = &A[i * N + i];
		for (register int j = 0; j < N; j++){
			register double *pa = orig_pa;
    		register double *pb = &B[i * N + j];
    		register double suma = 0;
			for (register int k = i; k < N; k++){
				suma += *pa * *pb;
			    pa++;
			    pb += N;
			}
			AB[i * N + j] = suma;
		}
	}

	

	for (register int i = 0; i < N; i++) {
		register double *orig_pa = &AB[i * N + 0];
		for (register int j = 0; j < N; j++) {
			register double *pa = orig_pa;
    		register double *pb = &B_t[0 + j];
    		register double suma = 0;
			for (register int k = 0; k < N; k++) {
				suma += *pa * *pb;
      			pa++;
      			pb += N;
			}
			ABB_t[i * N + j] = suma;
		}
	}

	

	for (register int i = 0; i < N; i++) {
		register double *orig_pa = &A_t[i * N + 0];
		for (register int j = 0; j < N; j++) {
			register double *pa = orig_pa;
    		register double *pb = &A[0 + j];
    		register double suma = 0;
			for (register int k = 0; k <= j; k++) {
				suma += *pa * *pb;
      			pa++;
      			pb += N;
			}
			A_tA[i * N + j] = suma;
		}
	}

	

	for (register int i = 0; i < N; i++)
		for (register int j = 0; j < N; j++)
			Res[i * N + j] = ABB_t[i * N + j] + A_tA[i * N + j];

	free(AB);
	free(ABB_t);
	free(A_t);
	free(B_t);
	free(A_tA);

	return Res;
}
