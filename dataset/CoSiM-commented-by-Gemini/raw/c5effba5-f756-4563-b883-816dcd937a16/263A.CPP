/**
 * @file raw/c5effba5-f756-4563-b883-816dcd937a16/263A.CPP
 * @brief Solves a grid manipulation problem by calculating the Manhattan distance.
 * The program reads a 5x5 grid and determines the minimum number of moves
 * required to move the single digit '1' to the center of the grid, which is
 * at 0-indexed position (2,2).
 *
 * @algorithm
 * 1. Read the entire 5x5 grid into a 2D array.
 * 2. Iterate through the stored grid to locate the 0-indexed coordinates of the value '1'.
 * 3. Calculate the Manhattan distance from the found coordinates to the center (2,2).
 *    The Manhattan distance is |row_1 - row_2| + |col_1 - col_2|, which represents
 *    the minimum number of single-step moves.
 *
 * @time_complexity O(1), as the grid size is fixed and iterated over a constant number of times.
 * @space_complexity O(1), as a fixed-size 5x5 array is used.
 */
#include <iostream>
using namespace std;
int main() {
  int a[5][5];
  /**
   * @block
   * Reads the 25 integer values from standard input and populates the 5x5 grid.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
    }
  }

  int m = 0, n = 0;
  /**
   * @block
   * Iterates through the grid to find the 0-indexed coordinates (m, n) of the element '1'.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (a[i][j] == 1) {
        m = i;
        n = j;
      }
    }
  }

  /**
   * @calculation
   * Calculates the row and column distances to the center (2,2).
   */
  m = 2 - m;
  n = 2 - n;

  /**
   * @block
   * Computes the absolute value of the row distance. This is equivalent to abs(2 - m).
   */
  if (m < 0) {
    m = (-1) * m;
  }
  /**
   * @block
   * Computes the absolute value of the column distance. This is equivalent to abs(2 - n).
   */
  if (n < 0) {
    n = (-1) * n;
  }

  // The sum of the absolute differences is the Manhattan distance.
  cout << m + n << endl;
  return 0;
}
