/**
 * @file raw/44261260-6065-409e-b0d5-35b59de4cc0b/263A.CPP
 * @brief Solution to Codeforces problem 263A - Beautiful Matrix.
 *
 * @algorithm
 * The problem requires finding the minimum number of moves to bring the single
 * '1' in a 5x5 matrix to the center at position (3,3). A move consists of
 * swapping adjacent rows or columns. The minimum number of moves is the
 * Manhattan distance between the current position of the '1' and the target
 * center position.
 *
 * Manhattan Distance = |x1 - x2| + |y1 - y2|
 *
 * The algorithm is as follows:
 * 1. A 5x5 matrix is read from standard input. The code uses a 6x6 array to
 *    facilitate 1-based indexing, a common practice in competitive programming.
 * 2. The code iterates through the matrix to find the coordinates (i, j) of the '1'.
 * 3. Once the '1' is found, the number of row moves and column moves are calculated.
 *    - Row moves: `abs(i - 3)`
 *    - Column moves: `abs(j - 3)`
 * 4. The total number of moves is the sum of the row and column moves.
 * 5. This sum is printed as the result.
 *
 * @complexity
 * - Time Complexity: O(1), since the matrix size is fixed at 5x5. The loops run
 *   a constant number of times (25 iterations).
 * - Space Complexity: O(1), as the matrix `data` has a fixed size (6x6).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // CntI: Stores the number of row moves required.
  // CntJ: Stores the number of column moves required.
  int CntI = 0, CntJ = 0;
  // data: 5x5 matrix to store input, with 1-based indexing.
  int data[6][6];

  /**
   * @brief Reads the matrix and finds the position of the '1'.
   *
   * This nested loop iterates through a 5x5 grid (from 1,1 to 5,5).
   * When the element '1' is found, it calculates the Manhattan distance
   * from its current position (i,j) to the center (3,3).
   */
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> data[i][j];
      // Pre-condition: An element is read into data[i][j].
      // Invariant: If '1' is found, CntI and CntJ will be updated to hold
      // the Manhattan distance components.
      if (data[i][j] == 1) {
        // Calculate the absolute difference in rows from the center (3).
        if (i < 3)
          CntI = 3 - i;
        else if (i > 3)
          CntI = i - 3;
        
        // Calculate the absolute difference in columns from the center (3).
        if (j < 3)
          CntJ = 3 - j;
        else if (j > 3)
          CntJ = j - 3;
      }
    }
  }

  // The total number of moves is the sum of row and column moves (Manhattan distance).
  cout << CntI + CntJ;
  return 0;
}
