/**
 * @file 50A.CPP
 * @brief A clear solution to the "Domino piling" problem (Codeforces 50A).
 *
 * @details This program calculates the maximum number of 2x1 dominoes that can be
 * placed on a rectangular board of size M x N. This version uses descriptive
 * variable names for clarity.
 *
 * @algorithm The solution is based on area calculation.
 * 1. The total area of the rectangular board is computed as `M * N`.
 * 2. The area of a single domino is 2.
 * 3. The maximum number of dominoes is the total area divided by the area of one
 *    domino. Integer division `area / 2` naturally finds the floor of this
 *    value, providing the correct answer for any board size.
 *
 * @complexity
 * Time Complexity: O(1), as the solution consists of a few arithmetic operations.
 * Space Complexity: O(1), as only a fixed number of variables are used.
 */
#include <iostream>

using namespace std;

int main() {
  int m, n, area;
  int areaOneDominoes = 2;

  cin >> m >> n;
  area = m * n;
  
  // Functional Logic: Calculate and print the maximum number of dominoes.
  cout << area / areaOneDominoes << endl;

  return 0;
}