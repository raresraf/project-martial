#include <bits/stdc++.h>
using namespace std;

/**
 * @file 50A.CPP
 * @brief Solution to the "Domino piling" problem from Codeforces (Problem 50A).
 * @details The program calculates the maximum number of 2x1 dominoes that can be
 * placed on a rectangular board of size M x N.
 *
 * Algorithm:
 * The total area of the rectangular board is M * N. Each domino has an area of 2x1 = 2.
 * The maximum number of dominoes that can fit on the board is the total area of the
 * board divided by the area of a single domino. Since we cannot use parts of a domino,
 * this is calculated using integer division.
 * Maximum Dominoes = floor((M * N) / 2).
 *
 * Time Complexity: O(1), as the solution involves a single arithmetic calculation.
 * Space Complexity: O(1).
 */
int main() {
  int n, m;
  // Read the dimensions of the board.
  cin >> n >> m;
  
  // Calculate and print the maximum number of dominoes.
  cout << (n * m) / 2;
  return 0;
}
