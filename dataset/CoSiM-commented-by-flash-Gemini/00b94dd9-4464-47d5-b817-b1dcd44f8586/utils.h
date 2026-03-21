/**
 * @00b94dd9-4464-47d5-b817-b1dcd44f8586/utils.h
 * @brief Centralized header file providing foundational utilities, abstract data types,
 *        and function prototypes crucial for the consistent implementation and
 *        rigorous testing of various matrix solver algorithms.
 *
 * This file serves as an interface for defining generic solver function pointers,
 * structuring test case parameters, and encapsulating common operations like
 * random number generation, thereby promoting modularity and reusability
 * across different solver implementations.
 */

#include <stdio.h>   // For standard I/O operations (e.g., FILE, printf)
#include <stdlib.h>  // For general utilities (e.g., malloc, free, rand, srand)
#include <sys/time.h> // For time-related functions (e.g., gettimeofday)

/**
 * @typedef Solver
 * @brief Defines a function pointer type for a matrix solver function.
 *
 * A function of this type is expected to take the matrix dimension `N` and
 * two pointers to `double` arrays representing square matrices `A` and `B`,
 * and return a pointer to a newly allocated `double` array representing the
 * result matrix. The caller is responsible for freeing the returned memory.
 *
 * @param N The dimension of the square matrices (N x N).
 * @param A A pointer to the first input matrix.
 * @param B A pointer to the second input matrix.
 * @return A pointer to the dynamically allocated result matrix.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @brief Macro to generate a random double-precision floating-point number within a symmetric range.
 *
 * Generates a random double value uniformly distributed within the range
 * `[-limit, +limit]`. Requires `rand()` to be seeded appropriately (e.g., with `srand()`).
 *
 * @param limit The positive maximum absolute value for the random number.
 * @return A random double value in `[-limit, limit]`.
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @struct test
 * @brief Aggregates configuration parameters for a single test case execution
 *        of a matrix solver, facilitating reproducible and configurable testing.
 */
struct test {
	int seed; /**< Functional Utility: Seed value for the pseudo-random number generator,
	                   ensuring that input matrices can be regenerated identically for
	                   reproducible test results across different runs. */
	int N;    /**< Functional Utility: Specifies the dimension (N x N) of the square matrices
	                   that will be used as input for the matrix solver in this test case. */
	/**
	 * Functional Utility: Buffer to store the file path where the output matrix generated
	 *                     by the solver for this specific test case should be saved.
	 * Rationale: This output file is crucial for later verification and debugging,
	 *            allowing comparison with known-good reference results.
	 */
	char output_save_file[100];
};

/**
 * @brief Prototype for the primary matrix solver function implemented in associated source files.
 *
 * Functional Utility: This function serves as the core computational engine,
 *                     taking two input matrices and performing a predefined
 *                     series of linear algebra operations to produce a result matrix.
 *                     Its specific implementation (e.g., optimized, non-optimized, BLAS-based)
 *                     is provided by the linked source files.
 *
 * @param N The dimension of the square matrices (N x N) for the computation.
 * @param A A pointer to the first input matrix, an array of `double` values in row-major order.
 * @param B A pointer to the second input matrix, an array of `double` values in row-major order.
 * @return A pointer to a newly dynamically allocated `double` array representing the
 *         result matrix. The caller is responsible for freeing this memory to prevent leaks.
 */
double* my_solver(int, double *, double *);

/**
 * @brief Prototype for a function designed to orchestrate and execute a single test case
 *        for a given matrix solver implementation.
 *
 * Functional Utility: This function encapsulates the entire testing workflow for a single scenario:
 *                     it initializes the test environment, invokes the specified `Solver` function
 *                     with generated input data, and measures its execution performance.
 *                     It is crucial for benchmarking and verifying the correctness of solver implementations.
 *
 * @param test_case A `struct test` containing all necessary parameters (seed, N, output file)
 *                  to configure the current test scenario.
 * @param solver_func A function pointer of type `Solver`, referencing the specific matrix
 *                    solver implementation that is to be tested.
 * @param total_time A pointer to a `float` variable where the measured execution time
 *                   (in seconds or milliseconds) of the `solver_func` for this test case will be stored.
 * @return An integer status code: 0 typically indicates successful test execution;
 *         non-zero values represent various failure conditions (e.g., memory allocation error,
 *         incorrect output).
 */
int run_test(struct test, Solver, float *);

/**
 * @brief Prototype for a utility function designed to safely deallocate dynamically
 *        allocated memory associated with matrix data.
 *
 * Functional Utility: This function is critical for proper memory management within
 *                     the matrix solver framework. It accepts a pointer to a pointer
 *                     to a `double` array, deallocates the memory pointed to by the
 *                     inner pointer, and then sets the inner pointer to `NULL` to
 *                     prevent dangling pointers.
 * Pre-condition: `data` points to a `double*` which in turn points to valid,
 *                dynamically allocated memory or is `NULL`.
 * Post-condition: The memory previously held by `*data` is freed, and `*data` is set to `NULL`.
 *
 * @param data A pointer to a pointer to the matrix data (a `double` array) to be freed.
 *             This allows the function to modify the caller's pointer to `NULL` after freeing.
 */
void free_data(double **);

/**
 * @brief Prototype for a function responsible for generating and populating matrices
 *        with random double-precision floating-point values for a given test case.
 *
 * Functional Utility: This function ensures that test scenarios can be executed
 *                     with varied yet reproducible input data. The `seed` from
 *                     `test_case` guarantees reproducibility, while the `is_input`
 *                     flag allows for flexible data generation strategies (e.g.,
 *                     populating input matrices vs. preparing a reference output matrix).
 *
 * @param test_case A `struct test` containing relevant parameters such as the `seed`
 *                  for random number generation and the matrix `N` for dimensions.
 * @param data A pointer to a pointer to `double`. Upon successful generation, this
 *             will point to a newly allocated `double` array containing the generated
 *             matrix data. The caller is responsible for freeing this memory.
 * @param is_input A flag (`int`, typically 0 or 1) indicating the purpose of the
 *                 generated data. A value of 1 might signify input matrices `A` or `B`,
 *                 while 0 might be used for other purposes, though its exact interpretation
 *                 depends on the implementation.
 * @return An integer status code: 0 for success, non-zero for failure (e.g., memory allocation error).
 */
int generate_data(struct test, double **, int);

/**
 * @brief Prototype for a function that parses an input file to extract
 *        configuration parameters for multiple test cases of matrix solvers.
 *
 * Functional Utility: This function automates the loading of test specifications,
 *                     enabling batch testing and consistent configuration across
 *                     different solver implementations and test runs. It dynamically
 *                     allocates memory to store the parsed test cases.
 *
 * @param file_name The path to the input configuration file (e.g., a text file
 *                  where each line defines parameters for a `struct test`).
 * @param num_tests A pointer to an `int` variable. Upon successful parsing,
 *                  this will store the total number of test cases read from the file.
 * @param tests A pointer to a pointer to `struct test`. Upon successful parsing,
 *              this will be updated to point to a newly dynamically allocated array
 *              of `struct test` structures, each populated with parameters from the file.
 *              The caller is responsible for freeing this array.
 * @return An integer status code: 0 for success, non-zero for failure (e.g., file not found,
 *         parsing error, memory allocation failure).
 */
int read_input_file(char *, int *, struct test **);
