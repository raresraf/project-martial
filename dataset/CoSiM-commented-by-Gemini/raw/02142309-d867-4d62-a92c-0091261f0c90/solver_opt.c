
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	double *C = malloc(N * N * sizeof(double));
	double *At = malloc(N * N * sizeof(double));
	double *Bt = malloc(N * N * sizeof(double));
	double *D = malloc(N * N * sizeof(double)); 
	double *E = malloc(N * N * sizeof(double)); 
	double *F = malloc(N * N * sizeof(double)); 

	if(C == NULL || At == NULL || Bt == NULL) {
		printf("Eroare la alocare");
		return NULL;
	}

	register int i;
	register int j;
	register int k;

	
	for(i = 0; i < N; i++) {
		register double *At_ptr = At + i; 
		register double *Bt_ptr = Bt + i; 

		register double *A_ptr = A + i * N; 
		register double *B_ptr = B + i * N; 

		for(j = 0; j < N; j++) {
			*At_ptr = *A_ptr;
			*Bt_ptr = *B_ptr;

			At_ptr += N;
			Bt_ptr += N;
			
			A_ptr++;
			B_ptr++;
		}
	}

	
	
	for(i = 0; i < N; i++) {
		register double *D_ptr = D + i * N; 

		for(j = 0; j < N; j++) {

			register double result = 0;

			register double *A_ptr = A + i * N;
			register double *Bt_ptr = Bt + j * N;

			for(k = 0; k < N; k++) {
				if(i <= k) {
					result += *(A_ptr + k) * *(Bt_ptr + k);
				}
			}

			*D_ptr = result;
			D_ptr++;
		}
	}

	
	
	for(i = 0; i < N; i++) {
		register double *E_ptr = E + i * N; 

		for(j = 0; j < N; j++) {

			register double result = 0;

			register double *D_ptr = D + i * N;
			register double *B_ptr = B + j * N;

			for(k = 0; k < N; k++) {
				result += *D_ptr * *B_ptr;
				D_ptr++;
				B_ptr++;
			}

			*E_ptr = result;
			E_ptr++;
		}
	}

	
	
	for(i = 0; i < N; i++) {
		register double *F_ptr = F + i * N; 

		for(j = 0; j < N; j++) {
			register double result = 0;

			register double *At_ptr1 = At + i * N;
			register double *At_ptr2 = At + j * N;

			for(k = 0; k < N; k++) {
				if(k >= i || k <= j) {
					result += *(At_ptr1 + k) * *(At_ptr2 + k);
				}
			}

			*F_ptr = result;
			F_ptr++;
		}
	}

	for(i = 0; i < N * N; i++) {
		register double *C_ptr = C + i;
		register double *E_ptr = E + i;
		register double *F_ptr = F + i;

		*C_ptr = *E_ptr + *F_ptr;
	}

	free(At);
	free(Bt);
	free(D);
	free(E);
	free(F);

	return C;
}
