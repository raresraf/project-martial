/**
 * @file 263A.CPP
 * @brief Finds the location of a '1' in a 5x5 grid and calculates its Manhattan distance to the center (2,2).
 *
 * This program solves a common competitive programming problem where the goal is to find the minimum
 * number of moves (up, down, left, right) to move a single '1' to the center of a 5x5 matrix.
 *
 * @algorithm Matrix Traversal and Manhattan Distance
 * @complexity
 * Time: O(1), as the matrix size is fixed at 5x5.
 * Space: O(1), for the 5x5 matrix storage.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @var arr The 5x5 integer matrix.
   * @var i, j Loop counters for iterating through the matrix.
   * @var xc, yc Coordinates (row and column) where the value '1' is found.
   */
  int arr[5][5], i, j, xc, yc;

  /**
   * @brief Reads the 5x5 matrix from standard input.
   *
   * This nested loop iterates through each cell of the matrix. When it
   * encounters the cell containing the value '1', it stores its coordinates
   * (i, j) in the variables xc and yc.
   */
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        xc = i;
        yc = j;
      }
    }
  }

  /**
   * @brief Calculates and prints the Manhattan distance to the center.
   *
   * The center of a 0-indexed 5x5 matrix is at (2, 2). The Manhattan distance
   * is the sum of the absolute differences of the coordinates, which represents
   * the minimum number of single-step moves (up, down, left, or right)
   * required to reach the center.
   */
  cout << abs(xc - 2) + abs(yc - 2) << endl;
  return 0;
}