/**
 * @file 263A.CPP
 * @brief Minimal move count calculation to achieve a matrix-centered state.
 * 
 * Logic: Identifies the position of a unique target element ('1') in a 5x5 grid 
 * and calculates its Manhattan Distance to the geometric center (2, 2) 
 * using 0-based indexing.
 * 
 * Algorithm: Single-pass scan followed by absolute displacement summation.
 * Time Complexity: $O(1)$ due to fixed input dimensions.
 * Space Complexity: $O(1)$ constant auxiliary space.
 */

#include <iostream>
using namespace std;

int main() {
  // State: grid coordinates of the target entity.
  int row, col;
  int num;

  /**
   * Block Logic: Exhaustive grid traversal.
   * Invariant: Grid contains exactly one '1' among twenty-five elements.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> num;
      // Functional Utility: Captures spatial indices of the target state.
      if (num == 1) {
        row = i;
        col = j;
      }
    }
  }

  /**
   * Functional Utility: Computes the sum of orthogonal displacements to center (2, 2).
   * Logic: Inline implementation of abs(row - 2) + abs(col - 2).
   */
  int steps = (row > 2 ? row - 2 : 2 - row) + (col > 2 ? col - 2 : 2 - col);
  
  // Output the total number of moves required.
  cout << steps;
  
  return 0;
}
