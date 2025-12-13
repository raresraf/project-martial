
#include "utils.h"


double *my_solver(int N, double *A, double *B) {
	printf("OPT SOLVER\n");

	register int i, j, k;
	register int size = N * N * sizeof(double);
	double *C, *At, *Bt, *AB;

	C = (double *) malloc(size);

	if (C == NULL)
		exit(-1);

	At = (double *) malloc(size);

	if (At == NULL)
		exit(-1);

	Bt = (double *) malloc(size);

	if (Bt == NULL)
		exit(-1);

	AB = (double *) malloc(size);

	if (AB == NULL)
		exit(-1);

	
	for (i = 0; i < N; ++i) {
		
		register double *pAt = At + i;
		
		register double *pBt = Bt + i;
		
		register double *pA = A + i * N;
		
		register double *pB = B + i * N;

		for (j = 0; j < N; ++j) {
			*pAt = *pA; *pBt = *pB;
			pAt += N; pBt += N;
			++pA; ++pB;
		}
	}

	
	for (i = 0; i < N; ++i) {
		
		register double *pAB = AB + i * N;
		
		register double *auxA = A + i * N;

		for (j = 0; j < N; ++j) {
			
			register double suma = 0;
			
			register double *pA = auxA + i;
			
			register double *pBt = Bt + j * N + i;

			for (k = i; k < N; ++k) {
				suma += *pA * *pBt;
				++pA; ++pBt;
			}

			*pAB = suma; ++pAB;
		}
	}

	
	for (i = 0; i < N; ++i) {
		
		register double *pC = C + i * N;
		
		register double *auxAB = AB + i * N;

		for (j = 0; j < N; ++j) {
			
			register double suma = 0;
			
			register double *pAB = auxAB;
			
			register double *pB = B + j * N;

			for (k = 0; k < N; ++k) {
				suma += *pAB * *pB;
				++pAB; ++pB;
			}

			*pC = suma; ++pC;
		}
	}

	
	for (i = 0; i < N; ++i) {
		
		register double *pC = C + i * N;
		
		register double *auxAt = At + i * N;

		for (j = 0; j < N; ++j) {
			
			register double suma = 0;
			
			register double *pAt = auxAt;
			
			register double *pAt2 = At + j * N;

			for (k = 0; k <= j; ++k) {
				suma += *pAt * *pAt2;
				++pAt; ++pAt2;
			}

			*pC += suma; ++pC;
		}
	}

	
	free(At);
	free(Bt);
	free(AB);

	return C;
}
