/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem.
 *
 * @details
 * The problem is to find the maximum number of 2x1 dominoes that can be placed
 * on a rectangular board of size M x N.
 *
 * Algorithm:
 * 1. The total area of the board is M * N.
 * 2. Each domino covers an area of 2.
 * 3. The maximum number of dominoes that can fit on the board is the total area
 *    divided by the area of a single domino.
 * 4. This is calculated using integer division: (M * N) / 2.
 *
 * Time Complexity: O(1), as it only involves a few arithmetic operations.
 * Space Complexity: O(1), as it uses a fixed amount of memory.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int a, b;
  // Read the dimensions of the rectangular board.
  cin >> a >> b;
  // Calculate the total area of the board and divide by the area of a domino (2)
  // using integer division to find the maximum number of dominoes that can fit.
  int res = (a * b) / 2;
  // Output the result.
  cout << res << endl;
}
