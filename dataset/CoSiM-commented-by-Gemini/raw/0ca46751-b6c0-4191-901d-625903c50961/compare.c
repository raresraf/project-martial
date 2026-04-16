/**
 * @file compare.c
 * @brief A utility to compare two matrices stored in binary files.
 *
 * This program takes three command-line arguments: two file paths pointing to
 * binary matrix data and a floating-point tolerance value. It memory-maps the
 * two files, interprets them as square matrices of doubles, and compares them
 * element-by-element to check if their absolute difference is within the
 * specified tolerance.
 *
 * This is primarily used as a correctness check for matrix solver benchmarks,
 * allowing the output of a test implementation to be compared against a known-good
 * or reference implementation's output.
 */
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>

/**
 * @def check_err(a, b, err)
 * @brief Checks if the absolute difference between two doubles is within a tolerance.
 * @param a The first double value.
 * @param b The second double value.
 * @param err The maximum allowed difference (tolerance).
 * @return 0 if the values are within tolerance, -1 otherwise.
 */
#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)


/**
 * @brief Compares the contents of two files, treating them as square matrices of doubles.
 *
 * This function opens and memory-maps two files. It first checks if their sizes are
 * identical. If so, it calculates the matrix dimension N (assuming a square matrix)
 * and iterates through all elements, comparing them using the `check_err` macro.
 * The comparison stops at the first mismatch.
 *
 * @param file_path1 Path to the first matrix file.
 * @param file_path2 Path to the second matrix file.
 * @param precision The floating-point tolerance for the comparison.
 * @return 0 if the matrices are identical within the given precision, -1 on error
 *         or if a mismatch is found.
 */
int cmp_files(char const *file_path1, char const *file_path2, double precision) {
	struct stat fileInfo1, fileInfo2;
	double *mat1, *mat2;
	int i, j, fd1, fd2, N, ret = 0;

	fd1 = open(file_path1, O_RDONLY, (mode_t)0600);
	fd2 = open(file_path2, O_RDONLY, (mode_t)0600);

	fstat(fd1, &fileInfo1);
	fstat(fd2, &fileInfo2);

	if(fileInfo1.st_size != fileInfo2.st_size) {
		printf("Files length differ
");
		close(fd1);
		close(fd2);
		return -1;
	}

	mat1 = (double*) mmap(0, fileInfo1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
	if (mat1 == MAP_FAILED)
	{
		close(fd1);
		close(fd2);
		printf("Error mmapping the first file");
		return -1;
	}

	mat2 = (double*) mmap(0, fileInfo2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
	if (mat2 == MAP_FAILED)
	{
		munmap(mat1, fileInfo1.st_size);
		close(fd1);
		close(fd2);
		printf("Error mmapping the second file");
		return -1;
	}

	N = sqrt(fileInfo1.st_size / sizeof(double));

	for (i = 0; i < N; i++ ) {
		for (j = 0; j< N; j++) {
			ret = check_err(mat1[i * N + j], mat2[i * N + j], precision); 
			if (ret != 0) {
				printf("Matrixes differ on index [%d, %d]. Expected %.8lf got %.8lf
",
						i, j, mat1[i * N + j], mat2[i * N + j]);
				goto done;
			}
		}
	}

done:
	munmap(mat1, fileInfo1.st_size);
	munmap(mat2, fileInfo2.st_size);

	close(fd1);
	close(fd2);

	return ret;
}

/**
 * @brief Main entry point for the matrix comparison utility.
 *
 * Parses command-line arguments for the two file paths and the comparison
 * tolerance, then calls `cmp_files` to perform the comparison. It prints "OK"
 * if the matrices match and "Incorrect results!" otherwise.
 *
 * @param argc The number of command-line arguments.
 * @param argv The command-line arguments:
 *             - argv[1]: Path to the first matrix file.
 *             - argv[2]: Path to the second matrix file.
 *             - argv[3]: Floating-point tolerance string.
 * @return 0 on success (matrices match), -1 on failure or mismatch.
 */
int main(int argc, const char **argv)
{    
	double precision;
	int ret = 0;

	if(argc < 4) {
		printf("Usage: %s mat1 mat2 tolerance
",argv[0]);
		exit(-1);
	}

	sscanf(argv[3], "%lf", &precision);

	ret = cmp_files(argv[1],argv[2],precision);
	
	printf("%s %s %s %s
", argv[0], argv[1], argv[2], (ret == 0 ? "OK" : "Incorrect results!"));
	return ret;
}
