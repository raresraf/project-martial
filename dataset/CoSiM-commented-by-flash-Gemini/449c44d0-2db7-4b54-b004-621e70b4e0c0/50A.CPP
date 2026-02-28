/**
 * @file 50A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 50A - "Domino piling").
 *
 * This program calculates the maximum number of 2x1 dominoes that can be placed
 * on a rectangular board of size M x N squares without overlapping.
 *
 * Algorithm: Simple arithmetic calculation.
 * The total number of squares on the board is N * M. Since each domino covers
 * exactly 2 squares, the maximum number of dominoes that can be placed is
 * (N * M) / 2, using integer division.
 *
 * Time Complexity: O(1), as it involves a constant number of arithmetic operations.
 * Space Complexity: O(1), as it only uses a few integer variables.
 */
#include "stdio.h" // Required for C-style input/output operations (scanf, printf).

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads the dimensions of a rectangular board and calculates the maximum number
 * of 2x1 dominoes that can be placed on it.
 */
int main() {
  int n, m; // 'n': number of rows, 'm': number of columns of the board.

  // Pre-condition: Read the two integers representing the dimensions of the board.
  scanf("%d%d", &n, &m);

  // Functional Utility: Calculate and print the maximum number of dominoes.
  // The total area is n*m, and each domino covers 2 squares. Integer division
  // automatically handles cases where the total area is odd (one square left uncovered).
  printf("%d\n", (n * m) / 2); // Added '\n' for cleaner output.

  return 0; // Indicate successful execution.
}
