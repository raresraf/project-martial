/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem.
 * @details The problem is to find the maximum number of 2x1 dominoes that can be placed on a rectangular board of size M x N.
 * The total area of the board is M * N. Each domino covers an area of 2.
 * Therefore, the maximum number of non-overlapping dominoes that can fit on the board is the total area divided by 2.
 *
 * @algorithm Area calculation.
 * @complexity
 * Time complexity: O(1), as it performs a single arithmetic calculation.
 * Space complexity: O(1).
 */
#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  // n, m: the dimensions of the rectangular board.
  int n, m;
  cin >> n >> m;
  // Calculate the total area of the board and divide by the area of a single domino (2).
  // Integer division automatically handles the floor of the result.
  int ans = (n * m) / 2;
  cout << ans;
  return 0;
}
