/**
 * @file 50A.CPP
 * @brief Solution for Codeforces Problem 50A - Domino piling.
 *
 * This program calculates the maximum number of 2x1 dominoes that can be
 * placed on a rectangular M x N board. Each domino covers exactly two squares.
 *
 * Algorithm: Simple area division. Since each domino covers two squares,
 * the maximum number of dominoes that can be placed on a board is half
 * of the total area of the board (M * N) if the area is even, or
 * (M * N - 1) / 2 if the area is odd (effectively floor(M*N / 2)).
 * Integer division in C++ automatically handles the floor operation.
 * Time Complexity: O(1) - constant time operations.
 * Space Complexity: O(1) - constant space usage.
 */

#include <cstdio> // Required for standard input/output functions like scanf and printf.
using namespace std; // Uses the standard namespace.

int main() {
  /**
   * @brief Main function to execute the Domino Piling problem logic.
   *
   * Reads the dimensions M and N of the rectangular board.
   * It then calculates the total area (M * N) and divides it by 2
   * using integer division to find the maximum number of dominoes that can be placed.
   * Finally, it prints this result.
   */
  // Declare integer variables for the dimensions of the board, M and N.
  int n, m;
  // Read the dimensions of the board. Assumes input order is M then N.
  scanf("%d%d", &m, &n);
  // Calculate the total area (m * n) and divide by 2 using integer division.
  // Integer division automatically floors the result, which is correct for this problem.
  printf("%d", (n * m) / 2);
  return 0; // Indicate successful program execution.
}
