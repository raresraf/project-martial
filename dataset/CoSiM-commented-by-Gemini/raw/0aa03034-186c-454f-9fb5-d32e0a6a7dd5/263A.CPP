/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem from competitive programming.
 * @details The goal is to find the minimum number of moves to bring the single '1' in a 5x5 grid to the center, which is at position (2,2) (0-indexed). A move consists of swapping adjacent rows or columns.
 * The number of moves required is the Manhattan distance from the '1's current position to the center.
 * This implementation first locates the '1', then simulates moving it one step at a time towards the center, counting each move.
 *
 * @algorithm Simulation of moves to calculate Manhattan distance.
 * @complexity
 * Time complexity: O(1), as the grid size is fixed at 5x5. The loops run a constant number of times.
 * Space complexity: O(1), for storing the 5x5 grid.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {

  // n: the 5x5 matrix.
  // m: an unused 1x1 matrix.
  // inrow, incol: the row and column index of the '1'.
  // cnt: a counter for the number of moves.
  int n[5][5], m[1][1], inrow, incol, cnt = 0;

  /**
   * @brief Reads the matrix and finds the position of the '1'.
   * @invariant Iterates through the 5x5 grid to find the element with value 1.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> n[i][j];
      if (n[i][j] == 1) {
        inrow = i;
        incol = j;
      }
    }
  }

  /**
   * @brief Simulates the moves to bring the '1' to the center.
   * @invariant In each iteration, this loop moves the '1' one step closer to the center (2,2) and increments the move counter.
   * The loop structure `for (int j = 0; j < 25; j++)` is a way to ensure termination, as the maximum number of moves is less than 25.
   */
  for (int j = 0; j < 25; j++) {
    // Moves the '1' horizontally towards the center column.
    if (incol > 2) {
      incol--;
      cnt++;
    } else if (incol < 2) {
      incol++;
      cnt++;
    // Once in the center column, moves the '1' vertically towards the center row.
    } else if (incol == 2 && inrow > 2) {
      inrow--;
      cnt++;
    } else if (incol == 2 && inrow < 2) {
      inrow++;
      cnt++;
    // If the '1' is at the center, print the count and exit.
    } else if (incol == 2 && inrow == 2) {
      cout << cnt << "\n";
      return 0;
    }
  }

  return 0;
}