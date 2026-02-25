/**
 * @file 50A.CPP
 * @brief Solution for a classic tiling problem from a programming contest.
 * @details The problem is to determine the maximum number of 2x1 dominoes that can
 * be placed on a rectangular board of size M x N without overlapping.
 *
 * Algorithm:
 * 1. The total area of the rectangular board is M * N.
 * 2. Each domino has an area of 2x1 = 2.
 * 3. The maximum number of non-overlapping dominoes that can fit on the board is
 *    the total area of the board divided by the area of a single domino.
 * 4. This can be calculated as `floor((M * N) / 2)`. Integer division in C++
 *    naturally performs the floor operation.
 *
 * Time Complexity: O(1), as the solution involves a single arithmetic calculation.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, m;
  cin >> m >> n;
  
  // The total area of the board is m * n. Each domino covers 2 squares.
  // Integer division correctly gives the maximum number of dominoes that can fit.
  cout << (m * n) / 2;
  
  return 0;
}