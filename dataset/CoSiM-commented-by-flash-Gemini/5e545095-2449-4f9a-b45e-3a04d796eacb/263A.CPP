/**
 * @5e545095-2449-4f9a-b45e-3a04d796eacb/263A.CPP
 * @brief This program solves problem 263A, "Beautiful Matrix". It reads a fixed 5x5 matrix
 * where exactly one cell contains the value 1. The goal is to determine the minimum
 * number of moves required to bring the cell containing 1 to the center of the matrix
 * (position (2,2) for 0-indexed coordinates). A move consists of shifting a row or column by one.
 * The calculation is equivalent to finding the Manhattan distance.
 *
 * Algorithm: Matrix traversal to locate a specific element, followed by direct Manhattan distance calculation
 * by summing pre-calculated row/column costs.
 * Time Complexity: O(1) as the matrix size (5x5) is constant.
 * Space Complexity: O(1) for storing the fixed-size matrix and a few variables.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes iostream.
using namespace std;     // Uses the standard namespace.

int main() {
  int a[5][5];       // Declares a 5x5 integer array (matrix).
  int i, j, m = 0;   // Declares loop counters (i, j), and 'm' (Manhattan distance accumulator) initialized to 0.
  
  /**
   * Block Logic: Reads the 5x5 matrix and calculates the Manhattan distance to the center (2,2)
   * when the cell containing 1 is found. The distance is accumulated in 'm'.
   * Invariant: After '1' is found, 'm' accumulates the absolute differences from the center.
   */
  for (i = 0; i < 5; i++) { // Outer loop iterates through rows (0 to 4).
    for (j = 0; j < 5; j++) { // Inner loop iterates through columns (0 to 4).
      cin >> a[i][j]; // Reads the integer value for the current cell.
      // Inline: If the current cell contains the value 1.
      if (a[i][j] == 1) { 
        // Block Logic: Adds the number of moves needed for the row to 'm'.
        // Rows 0 and 4 are 2 steps away from row 2. Rows 1 and 3 are 1 step away.
        if (i == 0 || i == 4) 
          m = m + 2;
        if (i == 1 || i == 3)
          m = m + 1;
        // Block Logic: Adds the number of moves needed for the column to 'm'.
        // Columns 0 and 4 are 2 steps away from column 2. Columns 1 and 3 are 1 step away.
        if (j == 0 || j == 4)
          m = m + 2;
        if (j == 1 || j == 3)
          m = m + 1;
      }
    }
  }
  cout << m << endl; // Prints the calculated total Manhattan distance.
  return 0;          // Indicates successful program termination.
}
