/**
 * @file 50A.CPP
 * @brief Solution for Codeforces Problem 50A, "Domino piling".
 * @details This program calculates the maximum number of 2x1 dominoes that can fit on an M x N board.
 * This version uses C-style I/O (scanf and printf) for input and output.
 *
 * @algorithm Mathematical Formula
 * @complexity
 * Time: O(1), as it's a single arithmetic calculation.
 * Space: O(1), using a fixed amount of memory.
 */
#include <cstdio> // For scanf and printf
// #include <bits/stdc++.h> // This is redundant when <cstdio> is used for I/O

using namespace std;

int main() {
  int n, m;

  // Read the board dimensions using C-style input.
  scanf("%d %d", &n, &m);

  /**
   * @brief Calculate and print the maximum number of dominoes.
   * The total area is n * m. Each domino covers 2 squares.
   * Integer division gives the maximum number of dominoes that can be placed.
   */
  printf("%d
", (n * m) / 2);

  return 0; // Indicate successful execution.
}