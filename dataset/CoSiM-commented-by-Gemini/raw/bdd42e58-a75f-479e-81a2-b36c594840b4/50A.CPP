/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem from Codeforces.
 * @details The program calculates the maximum number of 2x1 dominoes that can
 * be placed on a rectangular board of size M x N.
 *
 * Algorithm:
 * 1. Read the dimensions of the board, M and N.
 * 2. Calculate the total area of the board (grid = M * N).
 * 3. Each domino covers an area of 2.
 * 4. The maximum number of dominoes is the total area of the board divided by
 *    the area of a single domino. Integer division correctly handles cases
 *    where the total area is odd, as a half-domino cannot be placed.
 *
 * Time Complexity: O(1), as it involves a few arithmetic operations.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;
main() {
  // m, n: dimensions of the board.
  // grid: total area of the board.
  // dom: the maximum number of dominoes.
  int m, n, grid, dom;
  cin >> m >> n;
  grid = m * n;
  dom = grid / 2;
  cout << dom;
}
