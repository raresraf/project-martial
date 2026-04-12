/**
 * @file 263A.CPP
 * @brief Logic for calculating the minimal move count to achieve matrix-centered state.
 * 
 * Algorithm: Computes the Manhattan Distance from the target entity '1' to the 
 * geometric center (2, 2) of a 0-indexed 5x5 matrix.
 * 
 * Time Complexity: $O(1)$ due to fixed input dimensions.
 * Space Complexity: $O(1)$ constant memory for the grid state.
 */

#include <bits/stdc++.h>

using namespace std;

int main(void) {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams to 
   * optimize I/O performance.
   */
  ios_base::sync_with_stdio(0);
  cin.tie(0);

  // State: Grid coordinates of the target entity '1'.
  int oneI = 0;
  int oneJ = 0;
  // Simulation buffer for the 2D grid.
  int mat[5][5];

  /**
   * Block Logic: Exhaustive search to identify the coordinate of the target element.
   * Invariant: Grid contains exactly one '1' among twenty-five elements.
   */
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      cin >> mat[i][j];

      // Logic: Captures spatial indices upon identifying the target.
      if (mat[i][j] == 1) {
        oneI = i; 
        oneJ = j;
      }
    }
  }

  /**
   * Functional Utility: Computes the sum of displacements to the central index (2, 2).
   */
  cout << abs(oneI - 2) + abs(oneJ - 2) << "\n";

  return 0;
}
