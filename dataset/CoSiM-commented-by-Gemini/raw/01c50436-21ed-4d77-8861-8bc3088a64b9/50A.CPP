/**
 * @file 50A.CPP
 * @brief Solution to Codeforces problem 50A "Domino piling".
 *
 * @details
 * Problem statement: You are given a rectangular board of M × N squares. You have an unlimited
 * supply of standard dominoes of 2 × 1 squares. You are allowed to rotate the dominoes.
 * You are asked to place as many dominoes as possible on the board so as to meet the following
 * conditions:
 * 1. Each domino completely covers two squares.
 * 2. No two dominoes overlap.
 * 3. Each domino lies entirely inside the board. It is allowed to touch the edges of the board.
 * Find the maximum number of dominoes that can be placed.
 *
 * Algorithm: The total area of the board is M * N. Each domino covers an area of 2.
 * Therefore, the maximum number of dominoes that can be placed is the total area divided by the
 * area of a single domino, which is floor((M * N) / 2).
 *
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, m;
  scanf("%d %d", &n, &m);

  // The logic calculates the maximum number of 2x1 dominoes that can fit in an M x N grid.
  // This is equivalent to floor((m * n) / 2). The code implements this in a slightly
  // more complex way than necessary.

  // Calculate dominoes covering the main even part of the board.
  // It calculates how many pairs of rows can be filled completely.
  int i = m / 2;
  i *= n;

  // Handle the remaining single row, if M is odd.
  int x = m % 2, y = n / 2;
  if (x > 0 && y > 0)
    i += x * y; // Place remaining dominoes horizontally.
  
  printf("%d\n", i);
  return 0;
}