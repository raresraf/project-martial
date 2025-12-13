
#include "utils.h"
#include<stdint.h>




double* my_solver(int N, double *A, double* B) {

	register double* b_transpus = calloc(N * N, sizeof(double));
	register double* a_transpus = calloc(N * N, sizeof(double));

	register double* resultat = calloc(N * N, sizeof(double));
	register double* resultat2 = calloc(N * N, sizeof(double));
	register double* resultat3 = calloc(N * N, sizeof(double));
	register double* resultat4 = calloc(N * N, sizeof(double));

	for (register uint_fast32_t i = 0; i < N; ++i) {
		register double * ai = &A[i * N];
		register double * bi = &B[i * N];
		for (register uint_fast32_t j = 0; j < N; ++j) {
			register uint_fast32_t index = j * N + i;
		
			a_transpus[index] = *ai;
			b_transpus[index] = *bi;

			ai++;
			bi++;
		}
	}
	
	for (register uint_fast32_t i = 0; i < N; i++) {
		register uint_fast32_t p = i * N;
		register double *orig_pa = &A[p + i];
		for (register uint_fast32_t j = 0; j < N; j++) {
			register double *pa = orig_pa;
			register double *pb = &B[p + j] ;
			register double sum = 0.0;
			for (uint_fast32_t k = 0; k < N; k += 8) {
				 if(k >= i){
					sum += *pa * *pb;
					pa++;
					pb += N;
				}
				if(k + 1 >= i){
					sum += *pa * *pb;
					pa++;
					pb += N;
				}
				if(k + 2 >= i){
					sum += *pa * *pb;
					pa++;
					pb += N;
				}
				if(k + 3 >= i){
					sum += *pa * *pb;
					pa++;
					pb += N;
				}
				if(k + 4 >= i){
					sum += *pa * *pb;
					pa++;
					pb += N;
				}
				if(k + 5 >= i){
					sum += *pa * *pb;
					pa++;
					pb += N;
				}
				if(k + 6 >= i){
					sum += *pa * *pb;
					pa++;
					pb += N;
				}
				if(k + 7 >= i){
					sum += *pa * *pb;
					pa++;
					pb += N;
				}
			}
			resultat[p + j] = sum;
		}
	}

	register uint_fast32_t index;

	for (register uint_fast32_t i = 0; i < N; ++i) {
		register double *pAt_orig = &resultat[i * N];

		for (register uint_fast32_t j = 0; j < N; ++j) {
			register double suma1 = 0.0;

			register double *pAt = pAt_orig;
			register double *pB = &b_transpus[j];

			index = i * N + j;

			for (register uint_fast32_t k = 0; k < N; k += 8) {
				suma1 += *pAt * *pB;
				pAt++;
				pB += N;

				suma1 += *pAt * *pB;
				pAt++;
				pB += N;

				suma1 += *pAt * *pB;
				pAt++;
				pB += N;

				suma1 += *pAt * *pB;
				pAt++;
				pB += N;

				suma1 += *pAt * *pB;
				pAt++;
				pB += N;

				suma1 += *pAt * *pB;
				pAt++;
				pB += N;

				suma1 += *pAt * *pB;
				pAt++;
				pB += N;

				suma1 += *pAt * *pB;
				pAt++;
				pB += N;
			}
			resultat2[index] = suma1;

		}
	}

	for (register uint_fast32_t i = 0; i < N; ++i) {
		register double *pAt_orig = &a_transpus[i * N];

		for (register uint_fast32_t j = 0; j < N; ++j) {
			register double suma1 = 0.0;

			register double *pAt = pAt_orig;
			register double *pB = &A[j];

			index = i * N + j;

			for (register uint_fast32_t k = 0; k < N; k += 8) {
				if(i >= k){
					suma1 += *pAt * *pB;
				}
				pAt++;
				pB += N;

				if(i >= k + 1){
					suma1 += *pAt * *pB;
				}
				pAt++;
				pB += N;

				if(i >= k + 2){
					suma1 += *pAt * *pB;
				}
				pAt++;
				pB += N;

				if(i >= k + 3){
					suma1 += *pAt * *pB;
				}
				pAt++;
				pB += N;

				if(i >= k + 4){
					suma1 += *pAt * *pB;
				}
				pAt++;
				pB += N;

				if(i >= k + 5){
					suma1 += *pAt * *pB;
				}
				pAt++;
				pB += N;

				if(i >= k + 6){
					suma1 += *pAt * *pB;
				}
				pAt++;
				pB += N;

				if(i >= k + 7){
					suma1 += *pAt * *pB;
				}
				pAt++;
				pB += N;
			}
			resultat3[index] = suma1;

		}
	}	

	for(register uint_fast32_t i = 0; i < N * N; i += 8){
    	resultat4[i] = resultat3[i] + resultat2[i];
    	resultat4[i + 1] = resultat3[i + 1] + resultat2[i + 1];
    	resultat4[i + 2] = resultat3[i + 2] + resultat2[i + 2];
    	resultat4[i + 3] = resultat3[i + 3] + resultat2[i + 3];
    	resultat4[i + 4] = resultat3[i + 4] + resultat2[i + 4];
    	resultat4[i + 5] = resultat3[i + 5] + resultat2[i + 5];
    	resultat4[i + 6] = resultat3[i + 6] + resultat2[i + 6];
    	resultat4[i + 7] = resultat3[i + 7] + resultat2[i + 7];
    }

    free(resultat);
    free(resultat2);
    free(resultat3);
    free(a_transpus);
    free(b_transpus);

    return resultat4;

}

