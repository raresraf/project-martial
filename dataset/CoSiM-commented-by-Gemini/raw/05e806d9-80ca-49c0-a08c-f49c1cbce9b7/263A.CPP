/**
 * @file 263A.CPP
 * @brief Solution for Codeforces problem 263A, "Beautiful Matrix".
 *
 * @details This program calculates the minimum number of moves required to make a
 * 5x5 matrix "beautiful". A beautiful matrix is defined as one where the single
 * digit '1' is located at the center (intersection of the 3rd row and 3rd
 * column). A "move" consists of swapping two adjacent rows or two adjacent
 * columns.
 *
 * Algorithm:
 * The problem is equivalent to finding the Manhattan distance between the current
 * position of the '1' and the center of the matrix. The Manhattan distance
 * `|x1 - x2| + |y1 - y2|` represents the minimum number of single-step moves
 * (up, down, left, or right) to get from point (x1, y1) to (x2, y2) on a grid,
 * which directly corresponds to the row and column swaps.
 *
 * 1. Iterate through the 5x5 matrix to find the coordinates (row, col) of the '1'.
 * 2. The center of the matrix is at index (2, 2) (using 0-based indexing).
 * 3. Calculate the number of row swaps needed: `abs(row - 2)`.
 * 4. Calculate the number of column swaps needed: `abs(col - 2)`.
 * 5. The total minimum number of moves is the sum of these two values.
 *
 * Time Complexity: O(1), as the matrix size is fixed at 5x5. The loops run a
 * constant number of times (25).
 * Space Complexity: O(1), for storing the 5x5 matrix.
 */
// Beautiful Matrix
// https://codeforces.com/problemset/problem/263/A
//
//
// You've got a 5   5 matrix, consisting of 24 zeroes and a single number one.
// Let's index the matrix rows by numbers from 1 to 5 from top to bottom, let's
// index the matrix columns by numbers from 1 to 5 from left to right. In one
// move, you are allowed to apply one of the two following transformations to the
// matrix:
//
// Swap two neighboring matrix rows, that is, rows with indexes i and i + 1 for
// some integer i (1   i < 5). Swap two neighboring matrix columns, that is,
// columns with indexes j and j + 1 for some integer j (1   j < 5). You think
// that a matrix looks beautiful, if the single number one of the matrix is
// located in its middle (in the cell that is on the intersection of the third
// row and the third column). Count the minimum number of moves needed to make
// the matrix beautiful.
//
// Input
// The input consists of five lines, each line contains five integers: the j-th
// integer in the i-th line of the input represents the element of the matrix
// that is located on the intersection of the i-th row and the j-th column. It is
// guaranteed that the matrix consists of 24 zeroes and a single number one.
//
// Output
// Print a single integer   the minimum number of moves needed to make the
// matrix beautiful.
//
// middle=2,2
//
// 1,4
//
// 2,1

#include <iostream>
#include <math.h>
using namespace std;

int main() {

  int a[5][5];
  int ans = 0;
  // Iterate through each cell of the 5x5 matrix to find the '1'.
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {

      cin >> a[i][j];
      // When the '1' is found, calculate the Manhattan distance to the center.
      // The center is at index (2, 2).
      if (a[i][j] == 1)
        ans = abs(i - 2) + abs(j - 2);
    }
  }
  // Output the calculated minimum number of moves.
  cout << ans << endl;

  return 0;
}
