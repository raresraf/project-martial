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
 * Time Complexity: O(1) because the matrix size is fixed at 5x5. The string
 *                  manipulation for input might add a small constant factor,
 *                  but overall operations are constant.
 * Space Complexity: O(1) as a fixed-size array of strings is used, and it's 5x5.
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;
/**
 * @brief Main function to find the minimum moves for the Beautiful Matrix problem.
 *
 * Functional Utility: Reads the 5x5 matrix (represented unusually as strings),
 * locates the '1' element, and calculates the Manhattan distance to the center
 * (indices 2,2 for 0-indexed arrays).
 */
int main() {
  int i = 0, j = 0, a0, a1, a2, a3, a4; // Loop counters and temporary integer storage.
  long int row, col;                   // Variables to store the 0-indexed row and column of the '1'.
  string s[5];                         // Array of strings to store the matrix rows (unconventional storage).
  char tem[10];                        // Temporary buffer for sprintf.

  /**
   * Block Logic: Reads the 5x5 matrix elements. Each row is constructed as a string.
   * Simultaneously, it identifies and stores the 0-indexed row and column of the '1' element.
   * Pre-condition: Input stream contains 25 integers for the matrix.
   * Invariant: `row` and `col` store the coordinates of '1' after loops complete.
   */
  while (i < 5) { // Outer loop for rows (0 to 4).
    j = 0;
    s[i] = ""; // Initialize current row string.
    while (j < 5) { // Inner loop for columns (0 to 4).
      cin >> a0;      // Read an integer element.
      // Converts the integer to a string and appends it to the current row string.
      // This string representation is not directly used for the logic, only to store the matrix.
      sprintf(tem, "%d", a0);
      s[i] = s[i] + tem + " ";
      if (a0 == 1) { // If the current element is '1'.
        row = i;     // Store its row index.
        col = j;     // Store its column index.
      }
      j++; // Move to next column.
    }
    i++; // Move to next row.
  }
  // Calculate and print the Manhattan distance from (row, col) to the center (2,2).
  // fabs calculates the absolute value of a double, but works with int conversion.
  cout << fabs(row - 2) + fabs(col - 2);
  return 0;
}
