/**
 * @file 50A.CPP
 * @brief Solution for the "Domino piling" problem from Codeforces.
 * @details The problem asks for the maximum number of 2x1 dominoes that can be placed
 * on a rectangular board of size M x N without overlapping.
 *
 * @algorithm Area Division
 * The solution is a straightforward mathematical calculation.
 * 1. The total area of the board is `M * N`.
 * 2. Each domino covers an area of 2.
 * 3. The maximum number of dominoes is the total board area divided by the area of
 *    a single domino.
 * 4. Using integer division `(M * N) / 2` automatically floors the result, which correctly
 *    gives the maximum number of whole dominoes that can fit.
 *
 * @complexity
 * - Time: O(1), as it's a single arithmetic operation.
 * - Space: O(1).
 */

#include <iostream>

using namespace std;

int main() {
  // m and n are the dimensions of the rectangular board.
  int m, n;
  cin >> m >> n;

  /**
   * @brief Calculate and print the maximum number of dominoes.
   * The total area is `m * n`. Each domino has an area of 2.
   * Integer division `(m * n) / 2` gives the maximum number of
   * dominoes that can fit on the board.
   */
  cout << m * n / 2;
  
  return 0; // Indicate successful execution.
}
