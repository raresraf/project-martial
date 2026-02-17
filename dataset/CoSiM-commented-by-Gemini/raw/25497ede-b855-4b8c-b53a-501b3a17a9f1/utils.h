/**
 * @file utils.h
 * @brief Common interface, data structures, and prototypes for a matrix solver test harness.
 * @details This header defines a generic 'Solver' function pointer type, which allows
 * different solver implementations (e.g., naive, optimized, BLAS) to be used
 * interchangeably by a test framework. It also defines the data structure for a
 * test case and declares functions for running tests, generating data, and reading
 * configuration files.
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * @brief Defines a function pointer type 'Solver'.
 * Any function matching this signature can be used as a solver in the test framework.
 * It takes a matrix dimension N, and two input matrices A and B, and returns a
 * pointer to the resulting matrix C.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @brief Generates a random double within a specified range [-limit, limit].
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @struct test
 * @brief Holds all parameters required for a single test run.
 */
struct test {
	int seed; ///< The seed for the random number generator to ensure reproducibility.
	int N;    ///< The dimension of the square matrices to be used in the test.
	
	char output_save_file[100]; ///< Path to a file for saving the output.
};

/// @brief The common prototype for the solver function to be implemented in different files.
double* my_solver(int, double *, double *);

/// @brief Prototype for a function that runs a single test case with a given solver.
int run_test(struct test, Solver, float *);

/// @brief Prototype for a function to free allocated matrix data.
void free_data(double **);

/// @brief Prototype for a function that generates random matrix data for a test case.
int generate_data(struct test, double **, int);

/// @brief Prototype for a function that reads test configurations from an input file.
int read_input_file(char *, int *, struct test **);