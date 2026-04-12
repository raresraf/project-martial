/**
 * @file 263A.CPP
 * @brief Minimal move count calculation to center a specific element in a 5x5 matrix.
 * 
 * Algorithm: Computes the Manhattan Distance from the detected '1' coordinate 
 * to the geometric center (3, 3) using 1-based indexing.
 * 
 * Time Complexity: $O(1)$ due to fixed-size input (5x5 grid).
 * Space Complexity: $O(1)$ as it only stores iterative indices and the target location.
 */

#include <cmath>
#include <iostream>
using namespace std;

int main() {
  int i, j;
  int a;
  int flag = 0;

  /**
   * Block Logic: Sequential scan to find the unique '1' in the 5x5 grid.
   * Invariant: Search terminates immediately upon finding the target element (a == 1).
   */
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> a;
      // Functional Utility: Captures the precise grid coordinates of the target.
      if (a == 1) {
        flag = 1;
        break;
      }
    }
    // Optimization: Breaks the outer loop if the target was found in the inner loop.
    if (flag == 1)
      break;
  }

  /**
   * Functional Utility: Calculates the horizontal and vertical displacement to 
   * the center index (3, 3) for 1-based coordinates.
   */
  int res = abs(3 - i) + abs(3 - j);
  
  // Output the total distance (move count).
  cout << res;
  
  return 0;
}
