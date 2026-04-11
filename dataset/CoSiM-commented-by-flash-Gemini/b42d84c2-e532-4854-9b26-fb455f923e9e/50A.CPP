/**
 * @file 50A.CPP
 * @brief This program calculates the maximum number of dominoes (1x2) that can be placed
 * on a rectangular M x N board. Each domino covers two squares.
 *
 * Algorithm: Area division. The maximum number of dominoes is simply half the total area of the board.
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
/**
 * Subject: Domino piling
 * URL: https://codeforces.com/problemset/problem/50/A
 */
#include <iostream>

int main() {
  /**
   * Functional Utility: Optimizes standard input/output operations by unsynchronizing C++ streams
   * from C standard streams and unties cin from cout, improving performance for competitive programming.
   */
  std::ios::sync_with_stdio(0);
  std::cin.tie(0);
  std::cout.tie(0);

  // Functional Utility: 'm' and 'n' store the dimensions of the rectangular board.
  int m, n;
  // Block Logic: Reads the two dimensions of the board from standard input.
  std::cin >> m >> n;
  // Functional Utility: Calculates and prints the maximum number of dominoes.
  // Inline: The total area (m * n) is divided by 2, as each domino covers 2 squares.
  std::cout << (m * n) / 2 << "
";
  return 0;
}
