/**
 * @file solver_opt.c
 * @brief Manually optimized matrix solver implementation.
 * Features register blocking, loop reordering, and cache-friendly data access patterns.
 */

#include "utils.h"


/**
 * @brief Computes C = At * A + A * B * Bt.
 * Allocates memory dynamically and executes matrix operations.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
    printf("OPT SOLVER\n");
	double *C, *At, *Bt, *BBt; 
	register int i, j, k;
	
	C    = calloc(N * N, sizeof(*C));
	At   = calloc(N * N, sizeof(*At));
	Bt   = calloc(N * N, sizeof(*Bt));
	BBt  = calloc(N * N, sizeof(*BBt));

	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if((C == NULL) || (At == NULL) || (Bt == NULL) || (BBt == NULL)) {
        	return NULL;
    }
    
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; ++i) {
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; ++j) {
			/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
			if(i >= j)
				At[i * N + j] = A[j * N + i];
			Bt[i * N + j] = B[j * N + i];
		}
	}

	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i ++) {
        register double *orig_pa = &At[i * N]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
        /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
        for(j = 0; j < N; j ++) {
            register double *pa = orig_pa; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
            register double *pb = &A[j]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
            register double suma = 0;

            /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
            for(k = 0; k <= j; k++) {
	            suma += (*pa) * (*pb); /* Non-obvious pointer arithmetic/dereference for optimized memory access */
                pa ++; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
                pb += N; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
            }
            C[i * N + j] = suma;
        }
    }

    
    /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
    for (i = 0; i < N; i ++) {
        register double *orig_pa = &B[i * N]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
        /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
        for(j = 0; j < N; j ++) {
            register double *pa = orig_pa; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
            register double *pb = &Bt[j]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
            register double suma = 0;
            /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
            for(k = 0; k < N; k++) {
                suma += (*pa) * (*pb); /* Non-obvious pointer arithmetic/dereference for optimized memory access */
                pa ++; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
                pb += N; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
            }
            BBt[i * N + j] = suma;
        }
    }

    
    
    /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
    for (i = 0; i < N; i ++) {
        register double *orig_pa = &A[i * N]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
        /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
        for(j = 0; j < N; j ++) {
            register double *pa = orig_pa + i; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
            register double *pb = &BBt[j] + i * N; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
            register double suma = 0;
            /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
            for(k = i ; k < N; k++) {
                suma += (*pa) * (*pb); /* Non-obvious pointer arithmetic/dereference for optimized memory access */
                pa ++; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
                pb += N; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
            }
            C[i * N + j] += suma;
        }
    }

	free(At);
	free(Bt);
	free(BBt);
    
	return C;	
}
