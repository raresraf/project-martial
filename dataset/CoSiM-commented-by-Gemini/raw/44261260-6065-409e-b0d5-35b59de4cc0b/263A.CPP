/**
 * @file 263A.CPP
 * @brief Solution to Codeforces problem 263A - Beautiful Matrix.
 *
 * @algorithm
 * The problem asks for the minimum number of moves to make a 5x5 matrix "beautiful".
 * A beautiful matrix contains a single '1' at its center (row 3, column 3).
 * A move consists of swapping two adjacent rows or two adjacent columns.
 * The minimum number of moves required is the Manhattan distance between the
 * current position of the '1' and the center of the matrix.
 *
 * The algorithm is as follows:
 * 1. A 5x5 integer matrix is used, though the code declares a 6x6 one to
 *    conveniently use 1-based indexing.
 * 2. The code iterates through the 5x5 grid to read its elements.
 * 3. When the element '1' is found at position (i, j), its Manhattan distance
 *    to the center (3, 3) is calculated.
 * 4. The row distance is `abs(3 - i)` and the column distance is `abs(3 - j)`.
 *    The code implements this logic using if-else statements.
 * 5. These distances are stored in `CntI` and `CntJ`.
 * 6. The final result is the sum of `CntI` and `CntJ`.
 *
 * @complexity
 * - Time Complexity: O(1), since the matrix size is fixed at 5x5 (25 elements).
 *   The loops run a constant number of times.
 * - Space Complexity: O(1), as the matrix has a constant size.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // CntI: Stores the number of row moves needed (vertical distance).
  // CntJ: Stores the number of column moves needed (horizontal distance).
  int CntI = 0, CntJ = 0;
  // data: A 6x6 array to store the matrix, using 1-based indexing for a 5x5 grid.
  int data[6][6];
  /**
   * Block Logic: Iterate through the 5x5 grid to find the position of the '1'
   * and calculate its Manhattan distance from the center (3,3).
   */
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> data[i][j];
      // Pre-condition: An element is read into data[i][j].
      // Invariant: If '1' is found, CntI and CntJ will be updated to hold
      // the Manhattan distance components.
      if (data[i][j] == 1) {
        // Calculate the vertical distance to the center row (3).
        if (i < 3)
          CntI = 3 - i;
        else if (i > 3)
          CntI = i - 3;
        // Calculate the horizontal distance to the center column (3).
        if (j < 3)
          CntJ = 3 - j;
        else if (j > 3)
          CntJ = j - 3;
      }
    }
  }
  // Output the total Manhattan distance, which is the minimum number of moves.
  cout << CntI + CntJ;
  return 0;
}
