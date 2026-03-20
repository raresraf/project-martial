/**
 * @file 263A.CPP
 * @brief Solution for Codeforces problem 263A, "Beautiful Matrix".
 * @details This program calculates the minimum number of moves to bring a single '1'
 * in a 5x5 grid to the center, which is at position (2, 2) (0-indexed).
 *
 * @section Algorithm
 * The problem is solved by calculating the Manhattan distance between the current
 * position of the '1' and the center of the grid. The Manhattan distance
 * between two points (x1, y1) and (x2, y2) is given by |x1 - x2| + |y1 - y2|.
 *
 * 1. The code iterates through the 5x5 grid, reading each element.
 * 2. When the element '1' is found, its row and column indices (`row`, `col`) are stored.
 * 3. The final result is the Manhattan distance: `abs(row - 2) + abs(col - 2)`.
 *
 * Note: The code contains logic to construct strings for each row of the matrix,
 * but this is superfluous and not used in the final calculation.
 *
 * @subsection Complexity
 * - Time Complexity: O(1), as the grid size is fixed to 5x5, making the loops run a constant number of times.
 * - Space Complexity: O(1), as it uses a fixed amount of memory regardless of input.
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  // i, j are loop counters. a0-a4 are for input, though only a0 is used.
  int i = 0, j = 0, a0, a1, a2, a3, a4;
  // row, col will store the coordinates of the '1'.
  long int row, col;
  // s is an array of strings to store rows, tem is a char buffer for sprintf.
  // This string construction is not essential for the final answer.
  string s[5];
  char tem[10];

  /**
   * @brief This nested loop reads the 5x5 matrix to find the location of the '1'.
   * Pre-condition: Input stream contains 25 integer values for the 5x5 matrix.
   * Invariant: After the loops complete, `row` and `col` will hold the 0-indexed
   * coordinates of the element '1'.
   */
  while (i < 5) {
    j = 0;
    s[i] = "";
    while (j < 5) {
      cin >> a0;
      // The following two lines build a string representation of the row,
      // which is not used in the final computation.
      sprintf(tem, "%d", a0);
      s[i] = s[i] + tem + " ";
      if (a0 == 1) {
        row = i;
        col = j;
      }
      j++;
    }
    i++;
  }

  /**
   * @brief Calculate and output the Manhattan distance.
   * This is the number of single horizontal or vertical moves required to move the '1'
   * from its position (row, col) to the center (2, 2).
   */
  cout << fabs(row - 2) + fabs(col - 2);
  return 0;
}
