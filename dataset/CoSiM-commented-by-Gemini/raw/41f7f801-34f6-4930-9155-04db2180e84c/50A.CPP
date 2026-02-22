/**
 * @file 50A.CPP
 * @brief Solution to Codeforces problem 50A - Domino piling.
 *
 * @algorithm
 * The problem asks for the maximum number of 2x1 dominoes that can be placed on
 * a rectangular board of size M x N.
 *
 * The total area of the board is `M * N`. Each domino covers an area of 2
 * square units. Therefore, the maximum number of non-overlapping dominoes that
 * can fit on the board is the total area of the board divided by the area of a
 * single domino. Since dominoes cannot be broken, we use integer division to
 * find the result.
 *
 * The formula is simply `(M * N) / 2`.
 *
 * @complexity
 * - Time Complexity: O(1), as it's a direct calculation.
 * - Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int m, n;
  cin >> m >> n;
  // The total area of the board is m * n. Each domino covers an area of 2.
  // Integer division gives the maximum number of full dominoes that can fit.
  cout << (m * n) / 2 << endl;
  return 0;
}
