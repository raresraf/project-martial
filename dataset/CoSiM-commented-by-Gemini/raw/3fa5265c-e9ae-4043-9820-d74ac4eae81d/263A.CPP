/**
 * @file 263A.CPP
 * @brief Solution to Codeforces problem 263A - Beautiful Matrix.
 * 
 * @algorithm
 * The problem requires finding the minimum number of moves to bring a single '1'
 * in a 5x5 matrix to the center (position 3,3). A move consists of swapping
 * adjacent rows or columns. The minimum number of moves is equivalent to the
 * Manhattan distance between the current position of the '1' and the target
 * center position.
 * 
 * The algorithm is as follows:
 * 1. Iterate through the 5x5 matrix to find the coordinates (row `r`, column `c`) of the '1'.
 * 2. Calculate the Manhattan distance to the center (3,3) using the formula: |3 - r| + |3 - c|.
 * 3. Print the calculated distance.
 * 
 * @complexity
 * - Time Complexity: O(1), as the matrix size is fixed at 5x5. The loop runs a constant 25 times.
 * - Space Complexity: O(1), as it uses a few variables to store coordinates and the input value.
 */
// problem : https://codeforces.com/problemset/problem/263/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  int a, r, c;
  int i, j;
  /**
   * @brief Iterates through the 5x5 grid to locate the position of the '1'.
   * The loops are 1-indexed (1 to 5) to match the problem's coordinate system.
   */
  for (i = 1; i <= 5; i++)
    for (j = 1; j <= 5; j++) {
      scanf("%d", &a);
      if (a) {
        r = i;
        c = j;
      }
    }

  /**
   * @brief Calculates and prints the Manhattan distance.
   * This is the number of single row/column swaps required to move the '1' to the center (3,3).
   */
  printf("%d", abs(3 - r) + abs(3 - c));
}