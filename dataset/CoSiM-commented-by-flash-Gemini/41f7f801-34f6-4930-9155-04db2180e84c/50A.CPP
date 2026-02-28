/**
 * @file 50A.CPP
 * @brief Calculates the maximum number of dominoes that can be placed on a rectangular board.
 *
 * This program takes two integers, `m` and `n`, representing the dimensions
 * of a rectangular board (m rows, n columns). It then computes the maximum
 * number of 2x1 or 1x2 dominoes that can be placed on this board such that
 * no two dominoes overlap and each domino covers exactly two squares.
 *
 * Algorithm: Direct calculation using geometric principles. Since each domino
 *            covers exactly two squares, the maximum number of dominoes is
 *            simply half the total area of the board, i.e., `(m * n) / 2`.
 * Time Complexity: O(1) as it involves a constant number of arithmetic operations.
 * Space Complexity: O(1) as only a few scalar variables are used to store inputs and the result.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @brief Main function to read board dimensions and calculate max dominoes.
   *
   * Reads two integers `m` and `n` from standard input, representing the
   * dimensions of a rectangular board. Then calculates and prints the maximum
   * number of 2x1 dominoes that can be placed on this board.
   */
  // @param m Stores the number of rows of the rectangular board.
  // @param n Stores the number of columns of the rectangular board.
  int m, n;
  // Reads the dimensions m and n from standard input.
  cin >> m >> n;
  // @brief Calculates and prints the maximum number of dominoes.
  // Logic: Each domino covers 2 squares. So, max dominoes = total area / 2 = (m * n) / 2.
  cout << (m * n) / 2 << endl;
  return 0;
}
