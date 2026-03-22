/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem (Codeforces 50A).
 * @see https://codeforces.com/problemset/problem/50/A
 *
 * @details This program calculates the maximum number of 2x1 dominoes that can
 * be placed on a rectangular board of size M x N.
 *
 * @section algorithm
 * Algorithm: Area Calculation
 * The total area of the board is M * N. Each domino covers an area of 2.
 * The maximum number of non-overlapping dominoes that can fit on the board is
 * therefore the total area divided by the area of a single domino. Integer
 * division `(M * N) / 2` directly yields this result.
 *
 * @section complexity
 * - Time Complexity: O(1), as the solution is a single mathematical operation.
 * - Space Complexity: O(1), as it uses a fixed number of variables.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int m, n;
  cin >> m >> n;

  // The total area is m * n. Each domino has an area of 2.
  // Integer division gives the maximum number of full dominoes that can fit.
  cout << (n * m) / 2 << endl;
  
  return 0;
}
