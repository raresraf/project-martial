/**
 * @file 263A.CPP
 * @brief Solution to Codeforces problem 263A - Beautiful Matrix.
 * @description The problem involves a 5x5 matrix containing a single '1' and 24 '0's.
 *              The goal is to find the minimum number of moves required to move the '1'
 *              to the center of the matrix (position 3,3). A move consists of swapping
 *              two adjacent rows or two adjacent columns.
 *
 * Algorithm: The minimum number of moves is equal to the Manhattan distance between the
 *            current position of the '1' and the center of the matrix. The Manhattan
 *            distance between two points (r1, c1) and (r2, c2) is |r1 - r2| + |c1 - c2|.
 *            The program iterates through the matrix to find the coordinates of the '1'.
 *            Once found, it calculates the Manhattan distance to the center (3,3) and
 *            prints the result.
 *
 * Time Complexity: O(1), as the matrix size is fixed at 5x5. The program will always
 *                  perform at most 25 reads.
 * Space Complexity: O(1).
 */

// Implementation.

#include <bits/stdc++.h>

using namespace std;

int main() {
  int x = 0; // Variable to store the input value from the matrix.
  
  /**
   * Block Logic: Iterate through the 5x5 matrix to find the position of the '1'.
   * The loops use 1-based indexing (i from 1 to 5, j from 1 to 5) for easier
   * calculation with the center (3,3).
   */
  for (int i = 1; i <= 5; ++i) {
    for (int j = 1; j <= 5; ++j) {
      cin >> x;
      // Pre-condition: Check if the current element is the '1'.
      if (x == 1) {
        // Calculate the Manhattan distance from the current position (i, j)
        // to the center (3, 3) and print it. This is the minimum number of moves.
        cout << abs(i - 3) + abs(j - 3) << endl;
      }
    }
  }
  return 0;
}
