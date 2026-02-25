/**
 * @file 263A.CPP
 * @brief Solution for a grid problem, likely from a programming contest.
 * @details The program reads a 5x5 grid that contains a single '1' and all other
 * elements are '0'. It calculates the minimum number of moves (up, down, left, or right)
 * required to move the '1' to the center of the grid (position 3,3).
 *
 * Algorithm:
 * 1. Read the 5x5 grid element by element.
 * 2. Identify the row and column (coordinates) of the element that is equal to '1'.
 * 3. The minimum number of moves required to move an element from (x1, y1) to (x2, y2)
 *    in a grid using only axial movements is given by the Manhattan distance:
 *    |x1 - x2| + |y1 - y2|.
 * 4. Calculate the Manhattan distance between the coordinates of the '1' and the
 *    center of the grid, which is at (3,3) in a 1-indexed 5x5 grid.
 * 5. Print the result.
 *
 * Time Complexity: O(1), as the grid size is fixed to 5x5 (25 operations).
 * Space Complexity: O(1), as the storage is a fixed-size 5x5 array.
 */
#include <iostream>
#include <math.h> // For abs()
using namespace std;

int main() {
  int a[5][5], x, y;
  
  /**
   * @block
   * @description Reads the 5x5 grid and finds the 1-indexed coordinates (x, y) of the '1'.
   * The loops iterate through the matrix to locate the target element.
   */
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        x = i + 1;
        y = j + 1;
      }
    }
  }
  
  // Calculate and print the Manhattan distance to the center (3,3).
  cout << abs(x - 3) + abs(y - 3) << endl;
  return 0;
}
