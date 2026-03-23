/**
 * @file 50A.CPP
 * @brief Solution for Codeforces Problem 50A, "Domino piling".
 * @details This program calculates the maximum number of 2x1 dominoes that can
 * be placed on a rectangular board of size M x N.
 *
 * @algorithm Mathematical Formula
 * @complexity
 * Time: O(1), as it involves a single arithmetic calculation.
 * Space: O(1), using a fixed amount of memory.
 */

/*
Problem Statement:
Domino piling

You are given a rectangular board of M x N squares. Also you are given an
unlimited number of standard domino pieces of 2 x 1 squares. You are allowed to
rotate the pieces. You are asked to place as many dominoes as possible on the
board so as to meet the following conditions:

1. Each domino completely covers two squares.
2. No two dominoes overlap.
3. Each domino lies entirely inside the board. It is allowed to touch the edges
of the board.

Find the maximum number of dominoes, which can be placed under these
restrictions.
*/
#include <bits/stdc++.h> // Includes all standard libraries
using namespace std;

int main() {
  int n, m;

  // Read the dimensions of the board.
  cin >> n >> m;

  /**
   * @brief Calculate the maximum number of dominoes.
   * The total area of the board is n * m.
   * Each domino covers an area of 2 squares.
   * Therefore, the maximum number of non-overlapping dominoes is the total area
   * divided by the area of a single domino. Integer division handles cases
   * where one square might be left over.
   */
  cout << (n * m) / 2;

  return 0; // Indicate successful execution.
}