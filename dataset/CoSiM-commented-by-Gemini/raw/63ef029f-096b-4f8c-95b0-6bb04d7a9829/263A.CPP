/**
 * @file 263A.CPP
 * @brief Solution for the "Beautiful Matrix" problem.
 *
 * @problem A 5x5 matrix contains a single '1' and 24 '0's. The goal is to move
 * the '1' to the center of the matrix (row 3, column 3) using the minimum
 * number of moves, where a move consists of swapping two adjacent rows or two
 * adjacent columns.
 *
 * @algorithm The minimum number of moves required is the Manhattan distance between
 * the '1's initial position and the center of the matrix. The Manhattan distance
 * between two points (r1, c1) and (r2, c2) is |r1 - r2| + |c1 - c2|.
 * The algorithm iterates through the grid to find the coordinates (i, j) of the '1'
 * and then calculates the Manhattan distance to the center (3, 3).
 *
 * @time_complexity O(1), as the matrix size is fixed at 5x5.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // Use a 6x6 array to simplify 1-based indexing for the 5x5 matrix.
  int a[6][6], i, b, j;
  int n = 5;
  
  // Loop through the 5x5 grid using 1-based indexing (from 1 to 5).
  for (i = 1; i <= n; i++) {
    for (j = 1; j <= n; j++) {
      cin >> a[i][j];
      // When the '1' is found, calculate its Manhattan distance to the center (3, 3).
      if (a[i][j] == 1) {
        b = (abs(3 - i) + abs(3 - j));
      }
    }
  }
  
  // Print the calculated minimum number of moves.
  cout << b;
}
