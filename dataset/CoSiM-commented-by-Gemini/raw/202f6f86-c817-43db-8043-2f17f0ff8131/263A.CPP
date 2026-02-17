/**
 * @file 263A.CPP
 * @brief Solution for the "Beautiful Matrix" competitive programming problem.
 *
 * @details The problem involves a 5x5 matrix containing exactly one '1' and 24 '0's.
 * The goal is to find the minimum number of moves required to move the '1' to
 * the center of the matrix (position (2, 2) using 0-based indexing). A move
 * consists of swapping two adjacent rows or two adjacent columns.
 *
 * @section algorithm
 * Algorithm:
 * The minimum number of moves is the Manhattan distance between the current
 * position of the '1' and the center of the matrix.
 * 1. Read the 5x5 matrix from input.
 * 2. While reading, identify and store the row (`x`) and column (`y`) coordinates
 *    of the '1'.
 * 3. Calculate the number of row swaps needed as the absolute difference between
 *    the current row `x` and the target center row 2: `abs(x - 2)`.
 * 4. Calculate the number of column swaps needed as the absolute difference
 *    between the current column `y` and the target center column 2: `abs(y - 2)`.
 * 5. The total minimum moves is the sum of these two values.
 *
 * @section complexity
 * Time Complexity: O(1), as the input matrix size is fixed at 5x5. The program
 * performs a constant number of operations.
 * Space Complexity: O(1), for storing the fixed-size 5x5 matrix.
 */
#include <bits/stdc++.h>
using namespace std;

int matrix[5][5];

int main() {

  int i, j, x, y;
  // Block Logic: Read the matrix and find the coordinates of the '1'.
  // Pre-condition: The input contains 25 integers representing a 5x5 matrix.
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      scanf("%d", &matrix[i][j]);
      if (matrix[i][j] == 1) {
        x = i; // Store the row of the '1'.
        y = j; // Store the column of the '1'.
      }
    }
  }
  
  // Calculate the Manhattan distance from the current position (x, y) to
  // the center (2, 2) and print the result.
  printf("%d\n", (abs(x - 2) + abs(y - 2)));

  return 0;
}
