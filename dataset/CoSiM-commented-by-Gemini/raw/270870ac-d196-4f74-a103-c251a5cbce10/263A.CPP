/**
 * @file 263A.CPP
 * @brief Solution to the "Beautiful Matrix" problem (263A) from Codeforces.
 * @details The problem involves a 5x5 matrix containing a single '1' and twenty-four '0's.
 * The goal is to find the minimum number of moves (swapping adjacent rows or columns)
 * required to move the '1' to the center of the matrix, which is at position (3,3).
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm The solution is to find the Manhattan distance between the '1's current
 * position and the center of the matrix.
 * 1. Iterate through the 5x5 matrix to find the coordinates of the '1'.
 * 2. Store the 1-indexed row (`posY`) and column (`posX`) of the '1'.
 * 3. The minimum number of moves is the Manhattan distance between (`posX`, `posY`) and the
 *    center (3,3). The formula is `abs(posX - 3) + abs(posY - 3)`.
 * 4. Print the calculated distance.
 *
 * @complexity
 * Time Complexity: O(1) - The program iterates through a fixed-size 5x5 matrix, so the
 * number of operations is constant.
 * Space Complexity: O(1) - A constant amount of memory is used.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int posX, posY;
  /**
   * @block
   * @brief Reads the 5x5 matrix and finds the coordinates of the '1'.
   * @pre The input stream contains 25 integers representing the matrix. Exactly one integer is '1'.
   * @post `posX` and `posY` hold the 1-indexed column and row of the '1'.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      int inputan; // "inputan" is Indonesian for "input".
      scanf("%d", &inputan);
      if (inputan == 1) {
        // Store the 1-indexed position.
        posY = i + 1;
        posX = j + 1;
      }
    }
  }
  // Calculate and print the Manhattan distance to the center (3,3).
  printf("%d", abs(posX - 3) + abs(posY - 3));
}