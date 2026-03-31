/**
 * @file 50A.CPP
 * @brief Solves the "Domino Piling" problem.
 *
 * This program calculates the maximum number of 2x1 dominoes that can be placed
 * on a rectangular board of size M x N. The total area of the board is M * N,
 * and each domino covers an area of 2. The maximum number of dominoes is the
 * total area divided by 2, using integer division.
 *
 * @author f2eb56a3-4146-4c12-9d9c-8ee890820b84
 *
 * @algorithm Direct Calculation
 * @complexity
 * Time: O(1), as it involves a few basic arithmetic operations.
 * Space: O(1), for storing the input dimensions.
 */
#include <iostream>

using namespace std;

int main() {
  // `m` and `n` are the dimensions of the rectangular board.
  int m = 0, n = 0;
  // Read the dimensions from standard input.
  cin >> m >> n;
  /**
   * @brief Calculate and print the maximum number of dominoes.
   * The total area is `m * n`. Since each domino has an area of 2,
   * the maximum number of non-overlapping dominoes is `(m * n) / 2`
   * using integer division.
   */
  cout << (m * n) / 2 << endl;
  return 0;
}