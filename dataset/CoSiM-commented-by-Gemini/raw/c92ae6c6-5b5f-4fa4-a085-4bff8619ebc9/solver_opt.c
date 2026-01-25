


#include "utils.h"




void mul1 (double *A,double *B,double *C,int N){

int i,j,k;
double *pa,*pb,*pc;
double *orig_pc;


for(i = 0; i < N; i++){
	orig_pc = &C[i*N];
	pa=&A[i*N+i];
	for(k = i; k < N; k++){
		pc=orig_pc;
		register double cconst = *pa; 
		pa++;
		pb=&B[k*N];
		for(j = 0; j < N; j++){
			*pc += cconst * *pb;
			pc++;
			pb++;
			
			}
		}
	}

}


void mul2 (double *A,double *B,double *C,int N){

int i,j,k;
double *pa,*pb,*pc;
double *orig_pa;


pc=&C[0];
for(i = 0; i < N; i++){
	orig_pa = &A[i*N];
	pb=&B[0];
	for(j = 0; j < N; j++){
		pa=orig_pa;
		register double cconst = 0;
		for(k = 0; k < N; k++){
			cconst += *pa * *pb;
			pa++;
			pb++;
			
			}
		*pc=cconst;
		pc++;
	    }
	}



}


void mul3 (double *A,double *C,int N){

int i,j,k;
double *pa,*pb,*pc;
double *orig_pb;


for(i=0;i<N;i++){
pa=&A[i*N+i];
orig_pb=&A[i*N+i];
pc=&C[i*N];
	for(j=i;j<N;j++){
		pb=orig_pb;
		register double cconst = *pa;
		pa++;
		pc+=i;
		for(k=i;k<N;k++){
			*pc += cconst * *pb;
      		pc++;
      		pb++;
			
			}
		}
	}


}


void add(double *A,double *B,int N){

int i;
double *pa,*pb;
pa=&A[0];
pb=&B[0];

for(i=0;i<N*N;i++){
	*pa+=*pb;
	pa++;
	pb++;
	}

}


double* my_solver(int N, double *A, double* B) {

printf("OPT SOLVER\n");


double *C1=(double *)calloc(N*N,sizeof(double));
double *C2=(double *)calloc(N*N,sizeof(double));
double *C3=(double *)calloc(N*N,sizeof(double));


mul1(A,B,C1,N);

mul2(C1,B,C2,N);

mul3(A,C3,N);

add(C2,C3,N);



free(C1);
free(C3);

return C2;
}
