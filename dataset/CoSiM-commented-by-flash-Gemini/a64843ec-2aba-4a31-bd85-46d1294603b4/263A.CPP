/**
 * @file 263A.CPP
 * @brief Minimal move count calculation to achieve a matrix-centered state.
 * 
 * Algorithm: Computes the Manhattan Distance from the target entity '1' to the 
 * geometric center (2, 2) of a 0-indexed 5x5 matrix.
 * 
 * Time Complexity: $O(1)$ due to fixed input dimensions.
 * Space Complexity: $O(1)$ constant memory for the 5x5 grid state.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // State: x and y coordinates of the target entity '1'.
  int x, y, ans;
  // Simulation buffer for the 2D grid.
  int a[5][5];

  /**
   * Block Logic: Exhaustive search to identify the coordinate of the target element.
   * Invariant: Grid contains exactly one '1' among twenty-five elements.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
      // Logic: Captures spatial indices upon identifying the target state.
      if (a[i][j] == 1) {
        x = i;
        y = j;
      }
    }
  }

  /**
   * Functional Utility: Computes the sum of vertical and horizontal displacements 
   * to the central index (2, 2).
   */
  ans = abs(x - 2) + abs(y - 2);
  
  // Output the final count of required moves.
  cout << ans;
  
  return 0;
}
