/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem from Codeforces.
 * @author derekhh
 * @date May 31, 2015
 *
 * @details The problem asks for the maximum number of 2x1 dominoes that can be
 * placed on a rectangular board of size M x N without overlapping.
 *
 * @algorithm
 * The solution is a direct mathematical calculation. The total area of the
 * board is M * N. Each domino has an area of 2. By using integer division,
 * we can find the maximum number of whole dominoes that can fit on the board.
 * The formula is simply `(M * N) / 2`.
 *
 * @complexity
 * Time Complexity: O(1), as the solution involves a single arithmetic operation.
 * Space Complexity: O(1).
 */

// 50A.cpp
// A. Domino piling
// Author: derekhh
// May 31, 2015

#include <iostream>
using namespace std;

int main() {
  int m, n; // m and n are the dimensions of the rectangular board.
  cin >> m >> n;

  /**
   * @brief Calculate and print the maximum number of dominoes.
   * @logic The total number of squares on the board is `m * n`. Each domino
   * covers 2 squares. Integer division `(m * n) / 2` directly yields the
   * maximum number of non-overlapping dominoes that can be placed.
   */
  cout << m * n / 2 << endl;

  return 0;
}
