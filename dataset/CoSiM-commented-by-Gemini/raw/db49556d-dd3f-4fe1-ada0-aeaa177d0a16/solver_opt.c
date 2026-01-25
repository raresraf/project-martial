
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
	register double *ATA;
	register double *AB;
	register double *ABBT;
	int i, j, k;

	M = (double *)malloc(N * N * sizeof(double));
	DIE(M == NULL, "malloc M");

	ATA = (double *)malloc(N * N * sizeof(double));
	DIE(ATA == NULL, "malloc ATA");

	AB = (double *)malloc(N * N * sizeof(double));
	DIE(AB == NULL, "malloc AB");

	ABBT = (double *)malloc(N * N * sizeof(double));
	DIE(ABBT == NULL, "malloc ABBT");

	
	for (i = 0; i < N; i++) {
		register double *orig_pa = A + i * N;
		for (j = 0; j < N; j++) {
			register double *pa = orig_pa + i;
			register double *pb = B + j + i * N;
			register double suma = 0.0;
			for (k = i; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			AB[i * N + j] = suma;
		}
	}

	
	for (i = 0; i < N; i++) {
		register double *orig_pa = AB + i * N;
		for (j = 0; j < N; j++) {
			register double *pa = orig_pa;
			register double *pb = B + j * N;
			register double suma = 0.0;
			for (k = 0; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb++;
			}
			ABBT[i * N + j] = suma;
		}
	}

	
	for (i = 0; i < N; i++) {
		register double *orig_pa = A + i;
        	for (j = 0; j < N; j++) {
			register double *pa = orig_pa;
			register double *pb = A + j;
			register double suma = 0.0;
           		 for (k = 0; k < i + 1; k++) {
                		suma += *pa * *pb;
				pa += N;
				pb += N;
        		}
			ATA[i * N + j] += suma;
	    	}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			M[i * N + j] = ABBT[i * N + j] + ATA[i * N + j];
		}
	}

	free(ABBT);
	free(ATA);
	free(AB);

	return M;	
}
