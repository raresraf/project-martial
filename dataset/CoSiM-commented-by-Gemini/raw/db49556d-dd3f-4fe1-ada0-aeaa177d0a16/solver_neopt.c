
#include "utils.h"


#define DIE(assertion, call_description)				\
	do {								\
		if (assertion) {					\
			fprintf(stderr, "(%s, %d): ",			\
					__FILE__, __LINE__);		\
			perror(call_description);			\
			exit(EXIT_FAILURE);				\
		}							\
	} while (0)




 
 
double* my_solver(int N, double *A, double* B) {
	double *M;
	double *ATA;
	double *AB;
	double *ABBT;
	int i, j, k;

	M = (double *)calloc(N * N, sizeof(double));
	DIE(M == NULL, "calloc M");

	ATA = (double *)calloc(N * N, sizeof(double));
	DIE(ATA == NULL, "calloc ATA");

	AB = (double *)calloc(N * N, sizeof(double));
	DIE(AB == NULL, "calloc AB");

	ABBT = (double *)calloc(N * N, sizeof(double));
	DIE(ABBT == NULL, "calloc ABBT");
	
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				ABBT[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
        	for (j = 0; j < N; j++) {
           		 for (k = 0; k < i + 1; k++) {
                		ATA[i * N + j] += A[k * N + i] * A[k * N + j];
        		}
    		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			M[i * N + j] = ABBT[i * N + j] + ATA[i * N + j];
		}
	}	

	free(ATA);
	free(AB);
	free(ABBT);

	return M;
}
