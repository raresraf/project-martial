/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem from Codeforces.
 * @details The problem is to calculate the maximum number of 2x1 dominoes that
 * can be placed on a rectangular board of size M x N.
 *
 * @algorithm
 * The total area of the board is M * N. Each domino covers an area of 2.
 * The maximum number of dominoes is therefore floor((M * N) / 2), which is
 * equivalent to integer division `(M * N) / 2`.
 *
 * This implementation reaches the same result through case analysis:
 * 1. If N is even: The board can be perfectly tiled along the N-dimension.
 *    The number of dominoes is (N/2) * M.
 * 2. If M is even: The board can be perfectly tiled along the M-dimension.
 *    The number of dominoes is (M/2) * N.
 * 3. If both M and N are odd: The board can be tiled leaving one square empty.
 *    The number of dominoes is ((N*M) - 1) / 2. The code calculates this as
 *    `(N/2 * M) + (M/2)`.
 *
 * @complexity
 * Time Complexity: O(1), as it involves a constant number of arithmetic operations.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {

  int n, m; // Dimensions of the board.
  cin >> n >> m;

  int total = 0; // The result: max number of dominoes.

  /**
   * @block
   * @brief Calculate the number of dominoes based on the parity of the dimensions.
   * This case analysis is a more complex way to achieve integer division `(n * m) / 2`.
   */
  if (n % 2 == 0) {
    // If n is even, we can tile n/2 rows, each taking m dominoes.
    total = (n / 2) * m;
  } else if (m % 2 == 0) {
    // If m is even, we can tile m/2 columns, each taking n dominoes.
    total = (m / 2) * n;
  } else {
    // If both are odd, tile the largest possible even rectangle and then the remaining strip.
    // This is equivalent to ((n*m)-1)/2.
    total = ((n / 2) * m) + (m / 2);
  }

  cout << total << endl;

  return 0;
}
