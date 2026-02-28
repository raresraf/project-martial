/**
 * @file 263A.CPP
 * @brief Solution for the "Beautiful Matrix" problem from Codeforces.
 * @details The problem is to find the minimum number of moves (swapping adjacent
 * rows or columns) to move a single '1' in a 5x5 matrix to the center at (2,2).
 * This solution calculates the moves by checking the position of the '1' and
 * adding a pre-calculated number of moves based on its distance from the center.
 *
 * @algorithm Positional Move Calculation
 * 1. Iterate through the 5x5 matrix to find the coordinates (i, j) of the '1'.
 * 2. When the '1' is found, calculate the number of moves required based on its
 *    row and column index. This is equivalent to calculating the Manhattan distance
 *    to the center (2,2), but is done via a series of conditional checks:
 *    - If row is 0 or 4, it's 2 moves away from the center row.
 *    - If row is 1 or 3, it's 1 move away.
 *    - If column is 0 or 4, it's 2 moves away from the center column.
 *    - If column is 1 or 3, it's 1 move away.
 * 3. These moves are summed into a total, which is the final answer.
 *
 * @complexity
 * - Time: O(1), as the matrix size is fixed at 5x5, so the loops run 25 times.
 * - Space: O(1), for the fixed-size 5x5 matrix.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // `a` is the 5x5 matrix.
  int a[5][5];
  // `i` and `j` are loop counters.
  int i, j;
  // `m` will accumulate the total number of moves required.
  int m = 0;
  
  /**
   * @brief Read the matrix and calculate moves when the '1' is found.
   */
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> a[i][j];
      
      // Pre-condition: Check if the current element is the '1'.
      if (a[i][j] == 1) {
        /**
         * @brief Calculate the total moves (Manhattan distance) to the center (2,2).
         * This is done by adding the moves required for the row and the column separately.
         */
        
        // Add row moves. Rows 0 and 4 are 2 moves from the center (row 2).
        if (i == 0 || i == 4)
          m = m + 2;
        // Rows 1 and 3 are 1 move from the center.
        if (i == 1 || i == 3)
          m = m + 1;
        
        // Add column moves. Columns 0 and 4 are 2 moves from the center (col 2).
        if (j == 0 || j == 4)
          m = m + 2;
        // Columns 1 and 3 are 1 move from the center.
        if (j == 1 || j == 3)
          m = m + 1;
      }
    }
  }
  
  // Print the total accumulated moves.
  cout << m << endl;
  
  return 0; // Indicate successful execution.
}
