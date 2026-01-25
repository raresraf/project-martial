
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	double *C, *D, *A_t, *D_t;
	register int i, j, k;

	C = calloc(N * N, sizeof(double));
	if (C == NULL)
		exit(EXIT_FAILURE);

	D = calloc(N * N, sizeof(double));
	if (D == NULL)
		exit(EXIT_FAILURE);

	D_t = calloc(N * N, sizeof(double));
	if (D_t == NULL)
		exit(EXIT_FAILURE);
	
	A_t = calloc(N * N, sizeof(double));
	if (A_t == NULL)
		exit(EXIT_FAILURE);


	
	for (i = 0; i < N; i++) {
		
		register double *pD = D + i * N;
		for (j = 0; j < N; j++) {
			
			register double *pB = B + i * N;
			
			register double *pB_t = B + j * N;
			
			register double result = 0;
			for (k = 0; k < N; k++) {
				result += (*pB) * (*pB_t);
				pB++;
				pB_t++;
			}
			*pD = result;
			pD++;
		}
	}

	
	for (i = 0; i < N; ++i) {
		
		register double *pA = A + i * N;
		
		register double *pA_t = A_t + i;
		
		register double *pD = D + i * N;
		
		register double *pD_t = D_t + i;

		for (j = 0; j < N; j++) {
			*pA_t = *pA;
			
			pA++;
			
			pA_t += N;

			*pD_t = *pD;
			
			pD++;
			
			pD_t += N;
		}
	}

	
	for (i = 0; i < N; i++) {
		
		register double *pC = C + i * N;
		
		register double *pA_i_row = A + i * N;
		for (j = 0; j < N; j++) {
			
			register double result = 0;
			
			register double *pA = pA_i_row + i;
			register double *pD = D_t + j * N + i;
			for (k = i; k < N; k++) {
				result += (*pA) * (*pD);
				pA++;
				pD++;
			}
			*pC = result;
			pC++;
		}
	}

	
	
	
	
	for (i = 0; i < N; i++) {
		
		register double *pC = C + i * N;
		for (j = 0; j < N; j++) {
			
			register double *pA = A_t + i * N;
			
			register double *pA_t = A_t + j * N;
			
			register double result = 0;
			int min_index = i;
			if (min_index > j)
				min_index = j;
			for (k = 0; k <= min_index; k++) {
				result += (*pA) * (*pA_t);
				pA++;
				pA_t++;
			}
			*pC += result;
			pC++;
		}
	}


	free(D);
	free(A_t);
	free(D_t);

	return C;	
}
