/**
 * @file 263A.CPP
 * @brief Logic for determining the minimal move count to center a target element in a matrix.
 * 
 * Algorithm: Calculation of Manhattan Distance from the target state (1) to the 
 * geometric center of a fixed-size 2D lattice.
 * 
 * Time Complexity: $O(1)$ due to fixed matrix dimensions (5x5).
 * Space Complexity: $O(1)$ to store the matrix and coordinates.
 */

#include <iostream>

using namespace std;

int main() {
  // Static 5x5 grid representing the simulation space.
  int arr[5][5], row, col;

  /**
   * Block Logic: Exhaustive search to identify the coordinate of the target entity '1'.
   * Invariant: There is exactly one '1' in the 5x5 input stream.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> arr[i][j];
      // Logic: Captures the row and column indices of the target entity.
      if (arr[i][j] == 1) {
        row = i;
        col = j;
      }
    }
  }

  /**
   * Functional Utility: Computes the sum of vertical and horizontal displacements 
   * to the center point (index 2, 2).
   */
  cout << abs(2 - row) + abs(2 - col);
  
  return 0;
}
