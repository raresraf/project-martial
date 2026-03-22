/**
 * @file 263A.CPP
 * @brief Solution to Codeforces problem 263A - Beautiful Matrix.
 *        This program finds the minimum number of moves required to relocate the
 *        single '1' in a 5x5 matrix to the center position (row 3, column 3).
 *        A move consists of swapping an adjacent row or column.
 *
 * Algorithm: Matrix Traversal and Manhattan Distance Calculation
 *   1. Reads the elements of a 5x5 matrix.
 *   2. Identifies the row (`ai`) and column (`aj`) coordinates of the single '1' in the matrix.
 *      The loops break once the '1' is found.
 *   3. Calculates the Manhattan distance between the found coordinates (`ai`, `aj`)
 *      and the target center coordinates (3, 3) using the formula:
 *      `abs(target_row - current_row) + abs(target_col - current_col)`.
 *   4. Prints this calculated distance.
 *
 * Time Complexity: O(R * C), where R and C are the number of rows and columns of the matrix.
 *                  Since the matrix size is fixed at 5x5, this is effectively O(1).
 * Space Complexity: O(R * C) for storing the matrix. Effectively O(1) due to fixed matrix size.
 */

#include <bits/stdc++.h> // Includes common C++ standard library headers, including iostream and cmath for abs.
using namespace std;

int main() {
  /**
   * @brief Main function that solves the "Beautiful Matrix" problem.
   *
   * Reads a 5x5 matrix, finds the position of '1', and calculates the Manhattan
   * distance to move it to the center (3,3).
   *
   * @return int Returns 0 upon successful execution.
   */
  int i, j;           // Loop counters for rows and columns.
  int ai = 0, aj = 0; // Stores the row (`ai`) and column (`aj`) of the '1'.
  int a[6][6];        // 5x5 matrix, using 1-based indexing for convenience (size 6x6).
  
  /**
   * Block Logic: Iterates through the 5x5 matrix to find the position of the '1'.
   * Invariant: `a[i][j]` holds the current element being read. `ai` and `aj`
   *            store the coordinates of '1' once found.
   * Pre-condition: Input is a 5x5 matrix with a single '1' and other zeros.
   */
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> a[i][j]; // Reads the current element of the matrix.
      /**
       * Conditional Logic: If the current element is not zero, it must be the '1'.
       * Stores its coordinates and breaks the inner loop.
       */
      if (a[i][j] != 0) {
        ai = i; // Store row index.
        aj = j; // Store column index.
        break;  // Exit inner loop since '1' is found.
      }
    }
    // Note: If the '1' is found, the outer loop will continue, but `ai` and `aj`
    // will not be updated further since the '1' is unique and `break` would have
    // been executed from the inner loop, ensuring `ai` and `aj` hold the correct coordinates.
  }
  
  // Calculates and prints the Manhattan distance from the '1's position to the center (3,3).
  // `abs(3 - aj)` is the horizontal distance, `abs(3 - ai)` is the vertical distance.
  cout << (abs(3 - aj) + abs(3 - ai)) << "\n";
  return 0;
}
