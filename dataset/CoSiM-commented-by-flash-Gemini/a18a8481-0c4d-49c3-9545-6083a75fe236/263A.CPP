/**
 * @file 263A.CPP
 * @brief Solution for Codeforces Problem 263A - "Beautiful Matrix".
 * This program solves a problem involving a 5x5 matrix where one element is '1'
 * and all others are '0'. The goal is to find the position of the '1' and then
 * calculate the minimum number of moves required to bring this '1' to the center
 * of the matrix (position (2,2) for 0-indexed matrix) using only horizontal and
 * vertical movements. This is equivalent to calculating the Manhattan distance.
 *
 * @problem_statement
 * Given a 5x5 matrix, find the row and column of the element with value 1.
 * Calculate the Manhattan distance from this position to the center of the matrix (row 2, column 2).
 *
 * @algorithm
 * 1. Initialize variables `x` and `y` to store the row and column of the element '1'.
 * 2. Use nested loops to iterate through each cell of the 5x5 matrix (from row 0 to 4, column 0 to 4).
 * 3. Read the integer value `a` for the current cell.
 * 4. If `a` is 1, record its row `i` in `x` and its column `j` in `y`.
 * 5. After the loops complete (meaning the '1' has been found), calculate the Manhattan
 *    distance using the formula: `abs(x - 2) + abs(y - 2)`.
 * 6. Print the calculated distance.
 *
 * Time Complexity: O(1), as the matrix size is fixed at 5x5 (25 operations).
 * Space Complexity: O(1), as only a few fixed-size variables are used.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  /**
   * @brief Main function to find the '1' in a 5x5 matrix and calculate its Manhattan distance to the center.
   * Orchestrates reading matrix elements, identifying the target element, and computing the required moves.
   * @return 0 on successful execution.
   */
  long long i; ///< @var i Loop counter for rows (0-indexed).
  long long j; ///< @var j Loop counter for columns (0-indexed).
  long long x; ///< @var x Stores the row index of the '1' element.
  long long y; ///< @var y Stores the column index of the '1' element.
  long long dst; ///< @var dst Stores the calculated Manhattan distance.
  long long a; ///< @var a Temporary variable to store the current matrix element read from input.

  /**
   * @brief Nested loops to iterate through each cell of the 5x5 matrix.
   * @invariant The outer loop iterates through rows (`i`), and the inner loop iterates through columns (`j`).
   * `x` and `y` store the coordinates of the '1' once found.
   * @post `x` and `y` will contain the 0-indexed row and column of the element '1'.
   */
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> a; // Read the value of the current cell.
      /**
       * @brief Checks if the current cell contains the target value '1'.
       * @details If '1' is found, its coordinates are stored in `x` and `y`.
       * Since there is only one '1' in the matrix, this condition will be met exactly once.
       */
      if (a == 1) {
        x = i; // Store the row of '1'.
        y = j; // Store the column of '1'.
      }
    }
  }

  /**
   * @brief Calculates the Manhattan distance from the position of '1' to the center (2,2).
   * @details The Manhattan distance is calculated as `|x1 - x2| + |y1 - y2|`.
   * Here, (x1, y1) is (x, y) (position of '1') and (x2, y2) is (2, 2) (center of the 5x5 matrix).
   */
  dst = abs(x - 2) + abs(y - 2);
  cout << dst << endl; // Print the calculated distance.
  return 0;
}
