/**
 * @file 50A.CPP
 * @brief Solution to Codeforces Problem 50A: "Domino piling".
 *        This program calculates the maximum number of 2x1 dominoes that can be placed
 *        on a rectangular M x N chessboard, given that each domino covers two squares
 *        and dominoes cannot overlap.
 *
 * Algorithm: Mathematical formula based on the area of the board and parity of dimensions.
 *            The maximum number of dominoes is equivalent to (M*N)/2,
 *            since each domino covers 2 squares. The logic handles even/odd dimensions
 *            to correctly calculate this value.
 * Time Complexity: O(1) as it involves a few arithmetic operations.
 * Space Complexity: O(1) as only a few variables are used.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  /**
   * Functional Utility: Orchestrates the input of chessboard dimensions,
   *                     calculates the maximum number of dominoes, and outputs the result.
   */

  int n, m;      // Declares integers to store the dimensions (M and N) of the chessboard.
  cin >> n >> m; // Reads the dimensions of the chessboard from standard input.

  int total = 0; // Initializes a variable to store the maximum number of dominoes.

  /**
   * Block Logic: Calculates the maximum number of dominoes based on the parity of the chessboard dimensions.
   * Pre-condition: `n` and `m` represent the dimensions of the board.
   * Invariant: `total` will store the maximum number of 2x1 dominoes that can fit on the board.
   *            This logic essentially calculates (N * M) / 2, handling integer division and remainders correctly.
   */
  if (n % 2 == 0) {
    total = (n / 2) * m; // If N is even, half of N rows can be fully covered by dominoes.
  } else if (m % 2 == 0) {
    total = (m / 2) * n; // If M is even (and N is odd), half of M columns can be fully covered by dominoes.
  } else {
    // If both N and M are odd, one row of length M will have one square left,
    // and one column of length N will have one square left.
    // The solution covers (N-1)*M/2 + (M-1)/2, or simply (N*M - 1)/2.
    total = ((n / 2) * m) + (m / 2);
  }

  cout << total << endl; // Outputs the calculated maximum number of dominoes.

  return 0; // Indicate successful program termination.
}