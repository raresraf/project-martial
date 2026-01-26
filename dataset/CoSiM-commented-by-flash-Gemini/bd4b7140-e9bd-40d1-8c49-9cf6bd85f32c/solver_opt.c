
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	register int i, j, k;
    
	double *AtA;
	AtA = malloc(N * N * sizeof(double));

	double *D; 
	D = malloc(N * N * sizeof(double));

	double *C; 
	C = malloc(N * N * sizeof(double));
	
	
    register double *AtA_ind = AtA;
	for (i = 0; i < N; i++) {
        AtA_ind += i;
		for (j = i; j < N; j++) {
            register double elem = 0.0;
			register double *At_ind = A + i;
            register double *A_ind = A + j;
			for (k = 0; k < i+1; k++) {
                elem += *At_ind * (*A_ind);
                At_ind += N;
                A_ind += N;
			}
            *AtA_ind = elem;
            AtA_ind++;
		}
	}

	
	for (i = 1; i < N; i++) {
		register double *up_AtA = AtA + i * N;
		register double *down_AtA = AtA + i;
		
		for (j = 0; j < i; j++) {
			*up_AtA = *down_AtA;
			up_AtA++;
			down_AtA += N;

		}
	}

	
	register double *d = D;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			register double elem = 0.0;
			register double *a = A + i * N + i;
			register double *b = B + i * N + j;
			for (k = i; k < N; k++) {
				elem += *a * (*b);
				a++;
				b += N;
			}
			*d = elem;
			d++;
		}
		
	}

	
	register double *c = C;
    for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			register double elem = 0.0;
			register double *d_final = D + i * N;
			register double *b_final = B + j * N; 
			for (k = 0; k < N; k++) {
				elem += *d_final * (*b_final);
				d_final++;
				b_final++;
			}
			*c = elem;
			c++;
		}
	}


	
	register double *c_final = C;
	register double *ata_final = AtA;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			*c_final += *ata_final;
			c_final++;
			ata_final++;
		}

	}

	free(AtA);
	free(D);

	return C;
}
