
/**
 * @00b94dd9-4464-47d5-b817-b1dcd44f8586/compare.c
 * @brief Utility for robust, numerically tolerant comparison of two binary files
 *        representing square matrices of double-precision floating-point values.
 *
 * This program leverages memory-mapped I/O for high-performance data access and
 * conducts an element-wise comparison with a user-defined precision. Its primary
 * application is validating the output of matrix computation algorithms against
 * a trusted reference, crucial in scientific computing and performance-critical systems.
 *
 * Algorithm: Memory-mapped file I/O, element-wise comparison with floating-point tolerance.
 * Time Complexity: O(N^2) where N is the dimension of the square matrix.
 * Space Complexity: O(1) additional space beyond the memory-mapped files themselves,
 *                   as data is processed directly from mapped regions.
 */

#include <stdlib.h>  // For exit(), malloc(), free()
#include <stdio.h>   // For printf(), fprintf(), fopen(), fclose(), sscanf()
#include <stdint.h>  // For standard integer types (e.g., uint8_t)
#include <fcntl.h>   // For open()
#include <sys/stat.h> // For fstat()
#include <sys/mman.h> // For mmap(), munmap()
#include <unistd.h>  // For close()
#include <math.h>    // For fabs()

/**
 * @brief Macro to check if two double-precision floating-point numbers are approximately equal.
 *
 * This macro compares the absolute difference between two numbers (`a` and `b`)
 * against a given error tolerance (`err`).
 *
 * @param a The first double value.
 * @param b The second double value.
 * @param err The maximum allowed absolute difference (tolerance).
 * @return 0 if the absolute difference is less than or equal to `err`, indicating
 *         the numbers are considered equal; otherwise, returns -1.
 * Algorithm: Absolute difference comparison.
 */
#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)


/**
 * @brief Compares two binary files containing matrices of double-precision floating-point values.
 *
 * This function memory-maps the two input files and then performs an element-by-element
 * comparison. It assumes both files contain a square matrix of `double` values.
 * The comparison considers values equal if their absolute difference is within
 * the specified `precision`.
 *
 * @param file_path1 The path to the first binary file.
 * @param file_path2 The path to the second binary file.
 * @param precision The maximum allowed absolute difference between corresponding
 *                  elements for them to be considered equal.
 * @return 0 if the files contain identical matrices within the given precision,
 *         -1 if they differ or if an error occurs during file operations or memory mapping.
 */
int cmp_files(char const *file_path1, char const *file_path2, double precision) {
	struct stat fileInfo1, fileInfo2; /**< Functional Utility: Stores metadata (e.g., file size) for the first and second input files. */
	double *mat1, *mat2;             /**< Functional Utility: Pointers to the memory-mapped regions containing the double-precision matrix data from file1 and file2, respectively. */
	int i, j, fd1, fd2, N;           /**< Loop control variables (`i`, `j`), file descriptors (`fd1`, `fd2`), and computed matrix dimension (`N`). */
	int ret = 0;                     /**< Return value for the comparison result; initialized to 0 (success) and set to -1 on divergence or error. */

	/**
	 * Functional Utility: Opens the first binary file in read-only mode.
	 * Error Handling: In a robust production system, this would include checks for `fd1 < 0`.
	 */
	fd1 = open(file_path1, O_RDONLY, (mode_t)0600);
	/**
	 * Functional Utility: Opens the second binary file in read-only mode.
	 * Error Handling: In a robust production system, this would include checks for `fd2 < 0`.
	 */
	fd2 = open(file_path2, O_RDONLY, (mode_t)0600);

	/**
	 * Functional Utility: Retrieves detailed metadata (e.g., size) for the first opened file.
	 * Pre-condition: `fd1` must be a valid file descriptor.
	 */
	fstat(fd1, &fileInfo1);
	/**
	 * Functional Utility: Retrieves detailed metadata (e.g., size) for the second opened file.
	 * Pre-condition: `fd2` must be a valid file descriptor.
	 */
	fstat(fd2, &fileInfo2);

	/**
	 * Block Logic: Performs a critical preliminary check to ensure both files have identical sizes.
	 * Functional Utility: Verifies that the input files could potentially represent matrices of the same dimensions,
	 *                     a fundamental requirement for element-wise comparison. If sizes differ,
	 *                     comparison is impossible and an error is returned.
	 * Pre-condition: Both `fd1` and `fd2` are valid file descriptors, and `fileInfo1`, `fileInfo2` contain valid metadata.
	 * Error Handling: Prints an informative message to stderr and returns -1, indicating an uncomparable state.
	 */
	if(fileInfo1.st_size != fileInfo2.st_size) {
		fprintf(stderr, "Error: Files have different sizes. Cannot compare matrices.\n");
		close(fd1); // Resource Management: Close the first file descriptor to prevent leaks.
		close(fd2); // Resource Management: Close the second file descriptor to prevent leaks.
		return -1;  // Exit early due to irreconcilable difference.
	}

	/**
	 * Block Logic: Memory-maps the entire content of the first file into the process's virtual address space.
	 * Functional Utility: Enables zero-copy access to file data, significantly boosting I/O performance
	 *                     by eliminating redundant data transfers between kernel and user space.
	 * Pre-condition: `fileInfo1.st_size` is accurate and `fd1` is a valid, open file descriptor.
	 * Error Handling: If `mmap` fails, an error message is printed to stderr, and allocated file descriptors are closed before exiting.
	 */
	mat1 = (double*) mmap(0, fileInfo1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
	if (mat1 == MAP_FAILED)
	{
		close(fd1); // Resource Management: Close the first file descriptor.
		close(fd2); // Resource Management: Close the second file descriptor.
		fprintf(stderr, "Error: Failed to memory map the first file.\n");
		return -1; // Return error on mmap failure.
	}

	/**
	 * Block Logic: Memory-maps the entire content of the second file into the process's virtual address space.
	 * Functional Utility: Facilitates high-performance, element-wise comparison by allowing direct memory access
	 *                     to the second matrix's data, symmetric to the first file's mapping.
	 * Pre-condition: `fileInfo2.st_size` is accurate and `fd2` is a valid, open file descriptor.
	 * Error Handling: If `mmap` fails, an error message is printed to stderr, the previously mapped memory
	 *                 for `mat1` is unmapped, and all allocated file descriptors are closed before exiting.
	 * Resource Management: `munmap(mat1, fileInfo1.st_size)` ensures cleanup of previously acquired resources.
	 */
	mat2 = (double*) mmap(0, fileInfo2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
	if (mat2 == MAP_FAILED)
	{
		munmap(mat1, fileInfo1.st_size); // Resource Management: Unmap the first memory-mapped file.
		close(fd1); // Resource Management: Close the first file descriptor.
		close(fd2); // Resource Management: Close the second file descriptor.
		fprintf(stderr, "Error: Failed to memory map the second file.\n");
		return -1; // Return error on mmap failure.
	}

	/**
	 * Functional Utility: Derives the dimension `N` of the square matrix from the total file size.
	 * Calculation: Assumes the file contains N*N double-precision floating-point values.
	 *              `fileInfo1.st_size / sizeof(double)` gives the total number of elements.
	 *              The square root of this value yields `N`.
	 * Pre-condition: `fileInfo1.st_size` is a multiple of `sizeof(double)` and a perfect square
	 *                when divided by `sizeof(double)`.
	 */
	N = sqrt(fileInfo1.st_size / sizeof(double));

	/**
	 * Block Logic: Systematically traverses each corresponding element of the two memory-mapped matrices
	 *              to perform a numerically tolerant comparison.
	 * Algorithm: Element-wise comparison using nested loops. The comparison halts immediately upon
	 *            detection of the first significant difference, optimizing for early exit.
	 * Pre-condition: Both `mat1` and `mat2` point to valid memory-mapped matrix data, and `N` is the
	 *                correct dimension.
	 * Invariant: If the loop completes without `ret` becoming non-zero, the matrices are considered
	 *            identical within the specified `precision`.
	 */
	for (i = 0; i < N; i++ ) { // Iterates over rows of the matrices.
		for (j = 0; j< N; j++) { // Iterates over columns of the matrices.
			// Functional Utility: Compares corresponding elements `mat1[i][j]` and `mat2[i][j]`
			//                     using the `check_err` macro, which incorporates floating-point precision.
			// Pre-condition: `mat1[i * N + j]` and `mat2[i * N + j]` provide valid double values.
			ret = check_err(mat1[i * N + j], mat2[i * N + j], precision);
			/**
			 * Block Logic: Checks if the `check_err` macro detected a difference beyond the allowed precision.
			 * Functional Utility: Upon detecting a discrepancy, it logs the error with details
			 *                     and initiates an immediate jump to the cleanup phase (`goto done`).
			 */
			if (ret != 0) {
				fprintf(stderr, "Error: Matrices differ on index [%d, %d]. Expected %.8lf got %.8lf\n",
						i, j, mat1[i * N + j], mat2[i * N + j]);
				goto done; // Control Flow: Jumps to the `done` label to clean up resources and exit.
			}
		}
	}

done:
	/**
	 * Block Logic: Centralized resource cleanup mechanism. This section is reached either
	 *              upon successful completion of matrix comparison or prematurely
	 *              due to a detected difference.
	 * Functional Utility: Ensures that all system resources (memory maps, file descriptors)
	 *                     are properly released, preventing resource leaks.
	 * Pre-condition: `mat1`, `mat2` (if mapped) and `fd1`, `fd2` (if opened) are valid.
	 */
	munmap(mat1, fileInfo1.st_size); // Resource Management: Unmaps the memory region associated with the first matrix file.
	munmap(mat2, fileInfo2.st_size); // Resource Management: Unmaps the memory region associated with the second matrix file.

	close(fd1); // Resource Management: Closes the file descriptor for the first file.
	close(fd2); // Resource Management: Closes the file descriptor for the second file.

	return ret; // Functional Utility: Returns the final comparison result (0 for match, -1 for mismatch/error).
}

/**
 * @brief Main entry point for the matrix comparison utility.
 *
 * Parses command-line arguments: two file paths for the matrices and a
 * precision (tolerance) for floating-point comparison. It then calls
 * `cmp_files` to perform the comparison and prints a summary result.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of strings representing the command-line arguments.
 *             Expected format: `<executable_name> <matrix_file1> <matrix_file2> <tolerance>`.
 * @return 0 if the matrices are identical within the given precision,
 *         a non-zero value if they differ or if invalid arguments are provided.
 */
int main(int argc, const char **argv)
{
	double precision; /**< Functional Utility: Stores the floating-point tolerance value provided as a command-line argument, used for approximate comparison of matrix elements. */
	int ret = 0;      /**< Functional Utility: Holds the return status of the `cmp_files` function, indicating the outcome of the matrix comparison (0 for success, non-zero for failure). */
	/**
	 * Block Logic: Enforces correct command-line argument usage.
	 * Functional Utility: Checks if the minimum required number of arguments (executable name,
	 *                     two matrix file paths, and a tolerance value) are provided.
	 * Pre-condition: `argc` holds the count of command-line arguments.
	 * Error Handling: If `argc` is less than 4, a usage message is printed to `stderr`,
	 *                 and the program terminates with an error status, preventing
	 *                 further execution with invalid input.
	 */
	if(argc < 4) {
		fprintf(stderr, "Usage: %s mat1 mat2 tolerance\n", argv[0]);
		exit(-1); // Control Flow: Terminates the program due to invalid argument count.
	}

	/**
	 * Functional Utility: Extracts the numerical precision (tolerance) from the fourth
	 *                     command-line argument (`argv[3]`).
	 * Pre-condition: `argv[3]` is a valid string representation of a double-precision floating-point number.
	 */
	sscanf(argv[3], "%lf", &precision);

	/**
	 * Functional Utility: Invokes the core matrix comparison logic.
	 * Inputs: The file paths for the two matrices (`argv[1]`, `argv[2]`) and the parsed `precision`.
	 * Output: The return value (`ret`) indicates whether the matrices are identical within the given tolerance.
	 */
	ret = cmp_files(argv[1],argv[2],precision);
	
	/**
	 * Functional Utility: Displays a concise summary of the comparison outcome to standard output.
	 *                     This message includes the executable name, input file paths, and a status
	 *                     ("OK" for a match within tolerance, "Incorrect results!" otherwise).
	 */
	printf("%s %s %s %s\n", argv[0], argv[1], argv[2], (ret == 0 ? "OK" : "Incorrect results!"));
	return ret; /**< Functional Utility: Returns the integer status code of the matrix comparison
	                     to the operating system (0 for success, non-zero for failure or differences). */
}
