/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem from Codeforces.
 * @details The program calculates the maximum number of 2x1 dominoes that can
 * be placed on a rectangular board of size M x N.
 *
 * Algorithm:
 * 1. The total area of the board is M * N.
 * 2. Each domino covers an area of 2.
 * 3. The maximum number of dominoes is the total area of the board divided by
 *    the area of a single domino. Integer division handles the case where
 *    the total area is odd.
 *
 * Time Complexity: O(1), as it involves a few arithmetic operations.
 * Space Complexity: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  // m, n: dimensions of the rectangular board.
  int m, n; 
  cin >> m >> n;
  // The area of the board is m * n. Each domino has an area of 2.
  // The maximum number of dominoes is the integer division of the board's area by 2.
  cout << (m * n) / 2;
  return 0;
}
