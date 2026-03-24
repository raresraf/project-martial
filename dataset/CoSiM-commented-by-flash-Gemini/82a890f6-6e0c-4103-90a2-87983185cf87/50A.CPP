/**
 * @file 50A.CPP
 * @brief Solution to the "Domino Piling" competitive programming problem.
 *
 * This program calculates the maximum number of 2x1 dominoes that can be
 * placed on an M x N rectangular chessboard. Each domino covers exactly two squares.
 * The problem is a classic example of integer division in geometry/combinatorics.
 *
 * Algorithm: Direct calculation using integer division.
 * Time Complexity: O(1) as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) as it uses a constant amount of memory.
 */

#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.

/**
 * @brief Standard namespace for C++ functionalities.
 */
using namespace std;

/**
 * @brief Main function where the program execution begins.
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return int Returns 0 on successful execution.
 */
int main(int argc, char const *argv[]) {
  /**
   * @brief Optimizes C++ standard streams for faster input and output.
   * Functional Utility: Disables synchronization with C's stdio and unties cin from cout,
   *                     which is crucial for performance in competitive programming.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);

  int m, n;

  /**
   * @brief Reads the dimensions of the chessboard (M and N).
   *
   * Pre-condition: Two integer values are provided on standard input.
   * Post-condition: `m` and `n` store the respective dimensions.
   */
  cin >> m >> n;

  /**
   * @brief Calculates the total number of squares on the chessboard.
   * Functional Utility: Determines the area of the board.
   */
  int k = m * n;

  /**
   * @brief Calculates the maximum number of dominoes that can be placed.
   *
   * Functional Utility: Divides the total area by 2 (since each domino covers 2 squares).
   *                     Integer division naturally handles any remaining single square.
   * Algorithm: `(M * N) / 2`.
   */
  int ans = k / 2;

  /**
   * @brief Prints the calculated maximum number of dominoes.
   *
   * Pre-condition: `ans` holds the correct result.
   * Post-condition: The result is displayed on standard output.
   */
  cout << ans;

  return 0;
}