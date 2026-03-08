/**
 * @file 50A.CPP
 * @brief Calculates the maximum number of 2x1 dominoes that can be placed on an M x N rectangular board.
 *
 * Problem Name: Domino piling (Codeforces 50A)
 * Algorithm: Greedy approach. Each domino covers 2 squares, so the maximum number of dominoes is total squares / 2.
 * Time Complexity: O(1) due to constant time arithmetic operations.
 * Space Complexity: O(1) as it uses a fixed number of variables.
 */
// 50A.cpp
// A. Domino piling
// Author: derekhh
// May 31, 2015

#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

/**
 * @brief Main function to calculate the maximum number of dominoes.
 * @pre Input consists of two integers: 'm' (number of rows) and 'n' (number of columns).
 * @post Prints the maximum number of 2x1 dominoes that can be placed on an m x n board.
 */
int main() {
  int m, n; // 'm': number of rows, 'n': number of columns.

  // Functional Utility: Reads the dimensions of the board (m and n).
  cin >> m >> n;

  /**
   * Block Logic: Calculates the maximum number of dominoes using integer division.
   * Invariant: Total squares are m * n. Each domino covers 2 squares.
   *            Integer division by 2 naturally handles odd total squares by truncating the result,
   *            which correctly represents the maximum number of full dominoes.
   */
  cout << m * n / 2 << endl; // Prints the result.

  // Indicates successful program execution.
  return 0;
}