/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem from Codeforces contest 263.
 *
 * @details
 * The problem requires finding the minimum number of moves to make a 5x5 matrix "beautiful".
 * A matrix is considered beautiful if the single digit '1' it contains is located at the center,
 * which corresponds to the intersection of the 3rd row and 3rd column (index 2,2 in 0-based indexing).
 * A move consists of swapping any two adjacent rows or any two adjacent columns.
 *
 * The solution is to calculate the Manhattan distance between the position of the '1'
 * and the center of the matrix. The Manhattan distance, |x1 - x2| + |y1 - y2|, directly
 * corresponds to the minimum number of single row or column swaps required to move an element
 * from (x1, y1) to (x2, y2) on a grid.
 *
 * Algorithm:
 * 1. Iterate through the 5x5 grid, reading each element.
 * 2. When the element '1' is found, store its 0-indexed coordinates (row `x`, column `y`).
 * 3. Calculate the number of row swaps needed: `abs(x - 2)`.
 * 4. Calculate the number of column swaps needed: `abs(y - 2)`.
 * 5. The sum of these two values is the total minimum number of moves.
 *
 * Time Complexity: O(1), as the matrix size is fixed at 5x5, the loops run a constant 25 times.
 * Space Complexity: O(1), as only a few variables are used to store the coordinates and input.
 * The `a[110]` array is declared but not used in this specific problem.
 */
#include <bits/stdc++.h>
using namespace std;
int a[110]; // Unused array, likely a remnant from a template.
int main() {
  int x, y, e; // x: row of '1', y: column of '1', e: current element being read.
  
  // Block Logic: Iterates through each cell of the 5x5 matrix to find the location of the '1'.
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      cin >> e;
      // Pre-condition: Check if the current element is the target '1'.
      if (e == 1) {
        // Store the 0-indexed coordinates of the '1'.
        x = i;
        y = j;
      }
    }
  }
  
  // Functional Utility: Calculate the Manhattan distance from the '1's position (x, y)
  // to the center of the matrix (2, 2). This distance equals the minimum number of moves.
  cout << abs(x - 2) + abs(y - 2);
  
  return 0;
}