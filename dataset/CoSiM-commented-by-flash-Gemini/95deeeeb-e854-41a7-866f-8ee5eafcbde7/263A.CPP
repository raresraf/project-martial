/**
 * @file 263A.CPP
 * @brief Solution for Codeforces Problem 263A: Beautiful Matrix.
 * @details This program calculates the minimum number of moves required to bring the digit '1'
 * in a 5x5 matrix to the center position (row 2, column 2, assuming 0-indexed).
 * A move consists of shifting the '1' to an adjacent cell (up, down, left, or right).
 * This is equivalent to calculating the Manhattan distance between the current position of '1'
 * and the center position.
 * @algorithm Manhattan Distance Calculation.
 * Time Complexity: $O(1)$ since the matrix size is fixed (5x5).
 * Space Complexity: $O(1)$ to store the fixed-size matrix and a few variables.
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @file 263A.CPP
 * @brief Solution for Codeforces Problem 263A: Beautiful Matrix.
 * @details This program calculates the minimum number of moves required to bring the digit '1'
 * in a 5x5 matrix to the center position (row 2, column 2, assuming 0-indexed).
 * A move consists of shifting the '1' to an adjacent cell (up, down, left, or right).
 * This is equivalent to calculating the Manhattan distance between the current position of '1'
 * and the center position.
 * @algorithm Manhattan Distance Calculation.
 * Time Complexity: $O(1)$ since the matrix size is fixed (5x5).
 * Space Complexity: $O(1)$ to store the fixed-size matrix and a few variables.
 */
#include <bits/stdc++.h> // Standard C++ library includes (e.g., iostream, cmath for abs).

using namespace std; // Using the standard namespace to avoid prefixing std::.

int main() {
  /**
   * @brief Main function: orchestrates the matrix processing and distance calculation.
   */
  int m[5][5], // @var m: A 5x5 integer matrix.
      r,       // @var r: Stores the row index of the digit '1'.
      c;       // @var c: Stores the column index of the digit '1'.

  /**
   * @brief Reads the 5x5 matrix from standard input.
   * @block_logic Nested loops iterate through each cell of the 5x5 matrix to read its value.
   * @invariant `m[i][j]` contains the integer value read from input.
   */
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++)
      scanf("%d", &m[i][j]); // Reads an integer for each cell.

  /**
   * @brief Finds the coordinates (r, c) of the digit '1' in the matrix.
   * @block_logic Nested loops iterate through the matrix. The loop breaks once '1' is found.
   * @invariant After execution, `r` and `c` hold the 0-indexed row and column of the digit '1'.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (m[i][j] == 1) { // Checks if the current cell contains '1'.
        r = i;           // Stores the row index.
        c = j;           // Stores the column index.
        break;           // Exits the inner loop.
      }
    }
  }

  /**
   * @brief Calculates and prints the Manhattan distance to move '1' to the center (2,2).
   * @details The center of a 5x5 0-indexed matrix is at (2,2).
   * The Manhattan distance is `|x1 - x2| + |y1 - y2|`.
   * @block_logic Computes the absolute difference in rows and columns and sums them.
   * @invariant The printed value is the minimum moves required.
   */
  printf("%d\n", abs(2 - r) + abs(2 - c)); // Calculates and prints the Manhattan distance.
  return 0; // Indicates successful execution.
}