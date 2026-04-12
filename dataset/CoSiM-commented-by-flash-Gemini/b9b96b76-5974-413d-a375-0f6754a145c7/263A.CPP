/**
 * @file 263A.CPP
 * @brief Logic for determining the minimal move count to achieve a matrix-centered state.
 * 
 * Algorithm: Computes the Manhattan Distance from the target entity '1' to the 
 * geometric center (2, 2) of a 0-indexed 5x5 grid.
 * 
 * Logic: Performs a single-pass search and immediately outputs the calculated 
 * distance upon identifying the target state.
 * 
 * Time Complexity: $O(1)$ due to fixed input dimensions.
 * Space Complexity: $O(1)$ constant memory for grid storage.
 */

#include <iostream>
#include <math.h>
#include <vector>

using namespace std;

int main() {
  // Data Structure: 5x5 matrix representing the simulation grid.
  vector<vector<int>> input(5, vector<int>(5, 0));

  /**
   * Block Logic: Exhaustive search for the target element.
   * Invariant: Grid contains exactly one '1' among twenty-five cells.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> input[i][j];
      
      /**
       * Functional Utility: Conditional distance calculation.
       * Logic: Once the '1' is found at (i, j), the sum of orthogonal 
       * displacements to (2, 2) is calculated and emitted.
       */
      if (input[i][j] == 1) {
        cout << abs(2 - i) + abs(2 - j);
        // Optimization: Termination after finding the unique target.
        return 0;
      }
    }
  }
  
  return 0;
}
