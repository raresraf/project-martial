/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem on Codeforces.
 * @url https://codeforces.com/problemset/problem/263/A
 *
 * @details
 * The problem involves a 5x5 matrix containing a single '1' and the rest '0's.
 * The goal is to find the minimum number of moves to make the matrix "beautiful",
 * which means moving the '1' to the center of the matrix (at position (2,2)
 * using 0-based indexing). A move consists of swapping the '1' with an
 * adjacent cell (up, down, left, or right).
 *
 * @algorithm
 * The minimum number of moves required is the Manhattan distance between the
 * current position of the '1' and the center of the matrix. The Manhattan
 * distance between two points (x1, y1) and (x2, y2) is |x1 - x2| + |y1 - y2|.
 * In this case, the points are the position of '1' (x, y) and the center (2, 2).
 *
 * @complexity
 * Time Complexity: O(1), since the size of the matrix is fixed at 5x5, the
 * loops run a constant number of times.
 * Space Complexity: O(1), for storing the 5x5 matrix.
 */
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  int a[5][5], x, y, move;

  /**
   * @brief Read the matrix and find the position of the '1'.
   * The loops iterate through the 5x5 matrix to read its elements.
   * When the '1' is found, its row and column indices are stored in x and y.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        x = i;
        y = j;
      }
    }
  }

  /**
   * @brief Calculate the Manhattan distance to the center.
   * The center of the 5x5 matrix is at (2, 2) (0-indexed).
   * The number of moves is the sum of the absolute differences
   * in row and column indices.
   */
  move = abs(2 - x) + abs(2 - y);
  
  cout << move << endl;
}
