/**
 * @file 263A.CPP
 * @brief Solution to a problem involving finding a specific element in a 5x5 grid
 *        and calculating its Manhattan distance to the center.
 *
 * This program reads a 5x5 matrix (implicitly defined by nested loops and 'n=5').
 * It searches for the cell containing the value '1'. Once found, it calculates
 * the Manhattan distance from this cell's coordinates (i, j) to the center cell
 * (2, 2) of the grid. The Manhattan distance is defined as |x1 - x2| + |y1 - y2|.
 * The program then prints this calculated distance.
 * This is a common pattern in grid-based competitive programming problems
 * where one needs to find shortest paths or distances without diagonal movement.
 *
 * Algorithm: Brute-force search through a fixed-size 2D array, conditional
 *            Manhattan distance calculation. The search terminates upon first match.
 * Time Complexity: O(R * C), where R and C are the dimensions of the grid.
 *                  Since R=C=5, this is effectively O(1) as it's a constant-sized problem.
 * Space Complexity: O(1) for storing variables.
 */
#include <bits/stdc++.h> // Includes various standard libraries like iostream, cmath for abs.
using namespace std;     // Using the standard namespace.

int main() {
  // Functional Utility: Disables synchronization between C-style I/O and C++ streams
  // and unties cin from cout, speeding up input/output operations.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  // 'n' defines the dimension of the square grid (5x5). 'v' will store the value read from each cell.
  int n = 5, v;
  /**
   * Block Logic: Nested loops to iterate through each cell of a 5x5 grid (from (0,0) to (4,4)).
   * Invariant: 'i' represents the current row index, 'j' represents the current column index.
   *            The loops cover all cells of the grid.
   */
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      // Read the integer value for the current cell (i, j).
      cin >> v;
      /**
       * Block Logic: Checks if the current cell contains the target value '1'.
       * If 'v' is non-zero (implies v=1 as per problem context), calculate and print the distance.
       * Pre-condition: 'v' has been read from input.
       */
      if (v)
        // Calculate the Manhattan distance from current cell (i,j) to center (2,2).
        // |2 - i| is the absolute difference in rows, |2 - j| is the absolute difference in columns.
        // The program implicitly assumes there's only one '1' and exits after finding it.
        cout << abs(2 - i) + abs(2 - j) << "\n";
    }
  return 0; // Indicate successful program execution (implicitly if control reaches end of main).
}
