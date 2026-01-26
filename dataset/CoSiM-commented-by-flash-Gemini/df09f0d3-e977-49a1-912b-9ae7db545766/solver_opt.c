
#include "utils.h"
#include <string.h>


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	
	double *At = (double*) calloc(N * N, sizeof(double));
	double *Bt = (double*) calloc(N * N, sizeof(double));
	register int i, j, k;

	double *C1 = (double*) calloc(N * N, sizeof(double));

	
	for(i = 0; i < N; i++) {
		register double *matrix_lin = A + N * i;
		register double *matrix_col = At + i;

        for(j = 0; j < N; j++) {
			*matrix_col = *matrix_lin;
			matrix_lin++;
			matrix_col += N;
        }
    }

	
	for(i = 0; i < N; i++) {
		register double *matrix_lin = B + N * i;
		register double *matrix_col = Bt + i;

        for(j = 0; j < N; j++) {
			*matrix_col = *matrix_lin;
			matrix_lin++;
			matrix_col += N;
        }
    }

	
	for(i = 0; i < N; i++) {
	
	register double *pC = C1 + i * N;
	
	register double *orig_pA = A + i * N;

		for(j = 0; j < N; j++) {
			
			register double *pA = orig_pA + i;
			
			register double *pBt = Bt + j * N + i;
			
			register double suma = 0.0;

			for(k = i; k < N; k++) {
				suma += *pA * *pBt;
				pA++;
				pBt++;
			}

			*pC += suma;
			pC++;
		}
	}


	
	double *C2 = (double*) calloc(N * N, sizeof(double));

	for(i = 0; i < N; i++) {
		register double *pC = C2 + i * N;
		register double *orig_pC1 = C1 + i * N;

		for(j = 0; j < N; j++) {
			register double *pC1 = orig_pC1;
			register double *pB = B + j * N;
			register double suma = 0.0;

			for(k = 0; k < N; k++) {
				suma += *pC1 * *pB;
				pC1++;
				pB++;
			}

			*pC += suma;
			pC++;
		}
	}

	
	
	for(i = 0; i < N; i++) {
		register double *pC = C2 + i * N;
		register double *orig_At = At + i * N;

		for(j = 0; j < N; j++) {
			register double *pAt = orig_At;
			register double *pA = At + j * N ;
			register double suma = 0.0;

			for(k = 0; k <= i && k <= j; k++) {
				suma += *pAt * *pA;
				pAt++;
				pA++;
			}

			*pC += suma;
			pC++;
		}
	}

	free(At);
	free(Bt);
	free(C1);

	return C2;	
}
