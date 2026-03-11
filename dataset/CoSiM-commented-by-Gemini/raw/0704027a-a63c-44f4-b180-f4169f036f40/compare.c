
/**
 * @file compare.c
 * @brief A utility to compare two matrices stored in binary files.
 *
 * This program takes two file paths and a precision value as input. It reads the two
 * files, which are assumed to contain square matrices of doubles, and compares them
 * element by element to check if they are equal within the given precision.
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
 * @brief A macro that checks if two double-precision numbers are equal within a given tolerance.
 * @return 0 if the numbers are close enough, -1 otherwise.
 */
#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)

/**
 * @brief Compares two matrices stored in binary files.
 *
 * This function opens two files, memory-maps them, and compares their contents as
 * square matrices of doubles.
 *
 * @param file_path1 The path to the first file.
 * @param file_path2 The path to the second file.
 * @param precision The maximum allowed difference between corresponding elements.
 * @return 0 if the matrices are identical within the given precision, -1 otherwise.
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
 * @brief The main function of the program.
 *
 * It parses the command-line arguments and calls `cmp_files` to compare the two matrices.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 *             - argv[1]: Path to the first matrix file.
 *             - argv[2]: Path to the second matrix file.
 *             - argv[3]: The precision for the comparison.
 * @return 0 on success (matrices are equal), -1 on failure.
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
