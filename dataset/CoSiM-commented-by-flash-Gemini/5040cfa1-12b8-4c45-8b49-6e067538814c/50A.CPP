/**
 * @file 50A.CPP
 * @brief Solution for Codeforces Problem 50A - "Domino piling".
 * This program calculates the maximum number of 2x1 dominoes that can be placed on an M x N rectangular board.
 * Each domino covers two squares. The problem is a straightforward application of area division.
 * Algorithm: Direct calculation based on board area.
 * Time Complexity: O(1) - Involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) - uses a constant amount of memory for variables.
 */
#include <stdio.h> // Required for standard C-style input/output functions (scanf, printf).

int main(void) {
  int m; // Declares an integer to store the number of rows of the board.
  int n; // Declares an integer to store the number of columns of the board.

  // Reads the dimensions m and n of the board from standard input.
  scanf("%d %d", &m, &n);

  /**
   * Block Logic: Calculates the maximum number of dominoes based on the total area.
   * A 2x1 domino covers 2 square units. The maximum number of dominoes is the total
   * area divided by 2. Integer division correctly handles cases where the total area
   * is odd, leaving one square uncovered.
   */
  if (m * n % 2 == 0) { // If the total area (m * n) is even.
    printf("%d", m * n / 2); // The maximum dominoes is half the total area.
  } else { // If the total area (m * n) is odd.
    // The maximum dominoes is half of (total area - 1), effectively integer division.
    printf("%d", (m * n - 1) / 2); 
  }

  return 0; // Indicates successful program execution.
}
