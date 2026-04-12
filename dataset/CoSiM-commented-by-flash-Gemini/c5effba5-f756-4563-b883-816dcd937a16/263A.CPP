/**
 * @file 263A.CPP
 * @brief Solution to the "Beautiful Matrix" problem.
 *
 * This program calculates the minimum number of moves required to move the
 * element '1' in a 5x5 matrix to its center position. A move consists of
 * swapping adjacent rows or columns. The minimum moves correspond to the
 * Manhattan distance from the '1' to the center (index (2,2) for 0-indexed).
 *
 * Algorithm: Manhattan Distance Calculation.
 * Time Complexity: O(1) because the matrix size is fixed at 5x5, involving
 *                  a constant number of operations for input and calculation.
 * Space Complexity: O(1) as a fixed-size 2D array is used.
 */
#include <iostream>
using namespace std;
/**
 * @brief Main function to find the minimum moves for the Beautiful Matrix problem.
 *
 * Functional Utility: Reads the 5x5 matrix, locates the '1' element, and
 * calculates the Manhattan distance to the center (2,2) for 0-indexed arrays.
 */
int main() {
  int a[5][5]; // Declare a 5x5 integer matrix.

  /**
   * Block Logic: Reads the 5x5 matrix elements from standard input.
   * Pre-condition: Input stream contains 25 integers for the matrix.
   * Invariant: After these nested loops, `a` contains the initialized matrix.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j]; // Read element into matrix.
    }
  }

  int m = 0, n = 0; // Variables to store the 0-indexed row (m) and column (n) of the '1'.

  /**
   * Block Logic: Iterates through the 5x5 matrix to find the coordinates of the cell containing '1'.
   * Pre-condition: Matrix `a` is initialized with input values.
   * Invariant: After these nested loops, `m` and `n` store the 0-indexed coordinates of '1'.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (a[i][j] == 1) { // If the current element is '1'.
        m = i;           // Store its row index.
        n = j;           // Store its column index.
      }
    }
  }

  // Calculate the absolute difference from the center row (index 2).
  m = 2 - m;
  // Calculate the absolute difference from the center column (index 2).
  n = 2 - n;

  /**
   * Block Logic: Ensure the differences are absolute (positive).
   * Note: The `abs()` function from `<cmath>` could also be used.
   */
  if (m < 0) {
    m = (-1) * m;
  }
  if (n < 0) {
    n = (-1) * n;
  }

  cout << m + n << endl; // The sum of absolute differences is the Manhattan distance.
  return 0;
}
