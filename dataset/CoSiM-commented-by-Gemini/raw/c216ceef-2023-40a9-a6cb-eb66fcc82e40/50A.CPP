/**
 * @file 50A.CPP
 * @brief Solution for the "Domino Piling" problem (Codeforces 50A).
 * @details This program calculates the maximum number of 2x1 dominoes that can be placed
 * on a rectangular board of size M x N.
 *
 * Algorithm: Area Division
 * The problem is solved by a simple mathematical formula. The total area of the board is M * N.
 * Since each domino covers an area of 2, the maximum number of dominoes that can fit is
 * the total area divided by 2, using integer division.
 *
 * Time Complexity: O(1), as the solution involves a few arithmetic operations.
 * Space Complexity: O(1), as it uses a fixed amount of memory.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // m, n: dimensions of the rectangular board.
  // s: total area of the board.
  // t: declared but never used.
  // r: the result, which is the maximum number of dominoes.
  int m, n, s, t, r;

  // Read the dimensions of the board from standard input.
  cin >> m >> n;
  
  // Calculate the total area of the board.
  s = m * n;

  // The maximum number of 2x1 dominoes is the total area integer-divided by 2.
  r = s / 2;
  
  // Output the result.
  cout << r << endl;
  
  return 0;
}