/**
 * @file 50A.CPP
 * @brief Solution to the "Domino piling" competitive programming problem.
 * @details The problem asks for the maximum number of 2x1 dominoes that can be
 * placed on a rectangular board of size M x N without overlapping.
 *
 * @section algorithm Algorithm
 * The solution is based on a simple area calculation:
 * 1. The total area of the rectangular board is `M * N`.
 * 2. Each domino has an area of `2 * 1 = 2`.
 * 3. The maximum number of non-overlapping dominoes that can fit on the board
 *    is the total area of the board divided by the area of a single domino.
 * 4. Using integer division, `(M * N) / 2` correctly calculates `floor((M * N) / 2)`,
 *    which gives the maximum number of full dominoes that can be tiled.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(1), as the solution involves a fixed number of arithmetic operations regardless of input size.
 * - Space Complexity: O(1), as only a few variables are used to store the dimensions.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int m, n; // Board dimensions M and N.
  cin >> m >> n;
  
  // The total area is (m * n). Each domino covers 2 squares.
  // Integer division gives the maximum number of dominoes that can fit.
  long long int ans = (m * n) / 2;
  
  cout << ans << endl;
  return 0;
}