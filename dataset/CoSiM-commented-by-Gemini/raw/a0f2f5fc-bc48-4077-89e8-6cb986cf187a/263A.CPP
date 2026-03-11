/**
 * @file 263A.CPP
 * @brief Solution to the "Beautiful Matrix" problem (Codeforces 263A).
 * @details The problem asks for the minimum number of moves to bring a single '1'
 *          in a 5x5 matrix of zeros to the center at position (3, 3). A move
 *          consists of swapping two adjacent rows or two adjacent columns.
 *
 * @algorithm The minimum number of moves corresponds to the Manhattan distance
 *            between the current position of the '1' and the center of the matrix.
 *            The Manhattan distance between two points (r1, c1) and (r2, c2) in a
 *            grid is calculated as |r1 - r2| + |c1 - c2|.
 *            The algorithm finds the coordinates of the '1' and then calculates
 *            its Manhattan distance to the center (3, 3).
 *
 * @time_complexity O(1), as the matrix size is fixed at 5x5. The program will
 *                  at most perform 25 reads.
 * @space_complexity O(1), as it only stores the coordinates of the '1'.
 */
#include <cmath>
#include <iostream>
using namespace std;

int main() {
  int i, j;
  int a;
  int flag = 0;
  // The loops iterate through the 5x5 matrix to find the location of the '1'.
  // Using 1-based indexing for clarity, matching the problem's coordinates.
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> a;
      if (a == 1) {
        // When the '1' is found, set a flag and break the inner loop.
        flag = 1;
        break;
      }
    }
    // If the '1' was found in the inner loop, break the outer loop as well.
    if (flag == 1)
      break;
  }
  
  /**
   * @brief Calculate the Manhattan distance to the center (3, 3).
   * abs(3 - i) gives the number of row swaps needed.
   * abs(3 - j) gives the number of column swaps needed.
   */
  int res = abs(3 - i) + abs(3 - j);
  cout << res;
  return 0;
}