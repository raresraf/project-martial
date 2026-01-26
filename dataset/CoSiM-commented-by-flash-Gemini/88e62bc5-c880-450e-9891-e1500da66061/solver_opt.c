
#include "utils.h"



#define min(x, y) (((x) < (y)) ? (x) : (y))

double* my_solver(int N, double *A, double* B) {

        double *C = (double *)malloc(N * N * sizeof(double));
        double *term = (double *)malloc(N * N * sizeof(double));
	register double sum = 0.0;
	register double *orig_a, *a, *b;
	register int i, j, k, aux;

        for (i = 0;i < N;i++) {
		aux = i * N;
                orig_a = A + (aux + i);
		for (j = 0;j < N;j++) {
			a = orig_a;
			b = B + (aux + j);
                        sum = 0.0;
			for (k = i;k < N;k++) {
                                sum += *a * *b;
				a++;
				b += N;
                        }
			term[aux + j] = sum;
                }
        }

        for (i = 0;i < N;i++) {
		aux = i * N;
		orig_a = term + aux;
                for (j = 0;j < N;j++) {
			a = orig_a;
			b = B + (j * N);
                        sum = 0.0;
                        for (k = 0;k < N;k++) {
                                sum += *a * *b;
        			a++;
				b++; 
	               }  
			C[aux + j] = sum;
                }
        }

        for (i = 0;i < N;i++) {
        	aux = i * N;
		orig_a = A + i;     
		for (j = 0;j < N;j++) {
                        a = orig_a;
			b = A + j;
			sum = 0.0;
                        for (k = 0;k <= min(i, j);k++) {
                                sum += *a * *b;
                        	a += N;
				b += N;
			}
			term[aux + j] = sum;
		}
	}

	for (i = 0;i < N;i++) {
		aux = i * N;
		orig_a = term + aux;
                for (j = 0;j < N;j++) {
                        C[aux + j] += *orig_a;
			orig_a++;
                }
        }

        free(term);

	return C;	
}
