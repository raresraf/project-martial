/**
 * @63ef029f-096b-4f8c-95b0-6bb04d7a9829/263A.CPP
 * @brief This program solves problem 263A, "Beautiful Matrix". It reads a fixed 5x5 matrix
 * (using 1-indexed access in a 6x6 array for convenience) where exactly one cell contains
 * the value 1. The goal is to determine the minimum number of moves (Manhattan distance)
 * required to bring the cell containing 1 to the center of the matrix (position (3,3)
 * for 1-indexed coordinates).
 *
 * Algorithm: Matrix traversal to locate a specific element (the '1'), followed by
 * Manhattan distance calculation to the center.
 * Time Complexity: O(1) as the matrix size (5x5) is constant.
 * Space Complexity: O(1) for storing the fixed-size matrix and variables.
 */
#include <bits/stdc++.h> // A common header in competitive programming; includes iostream, cmath.
using namespace std;     // Uses the standard namespace.

/**
 * @brief Main function to find the '1' in a 5x5 matrix and calculate moves to center.
 * Reads the matrix, finds the coordinates of '1', and computes the Manhattan distance
 * to the center (3,3) (1-indexed).
 * @return 0 on successful execution.
 */
int main() {
  int a[6][6], i, b, j; // Declares a 6x6 integer array (matrix) for 1-indexed access,
                         // loop counters (i, j), and 'b' to store the Manhattan distance.
  int n = 5;             // Defines the effective size of the matrix (5x5).
  
  /**
   * Block Logic: Reads the 5x5 matrix and finds the coordinates of the cell containing the value 1.
   * While reading, it directly calculates the Manhattan distance to the center (3,3)
   * for the cell that contains '1'.
   * Invariant: After this nested loop, 'b' stores the Manhattan distance from the cell
   * containing '1' to the center of the matrix.
   */
  for (i = 1; i <= n; i++) { // Outer loop iterates through rows (1 to 5).
    for (j = 1; j <= n; j++) { // Inner loop iterates through columns (1 to 5).
      cin >> a[i][j]; // Reads the integer value for the current cell.
      if (a[i][j] == 1) { // Inline: If the current cell contains the value 1.
        // Calculates Manhattan distance: |row_of_1 - center_row| + |col_of_1 - center_col|.
        // Center for 1-indexed 5x5 is (3,3).
        b = (abs(3 - i) + abs(3 - j)); 
      }
    }
  }
  cout << b; // Prints the calculated Manhattan distance.
}
