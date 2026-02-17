/**
 * @file 50A.CPP
 * @brief Solution for the "Domino Piling" problem, likely Codeforces 50A.
 * @details This program calculates the maximum number of 2x1 dominoes that can
 * be placed on a rectangular board of given dimensions M x N.
 *
 * @section
 *
 * Algorithm:
 * The solution is based on a simple mathematical formula. The total area of the
 * board is M * N. Each domino covers an area of 2. The maximum number of
 * non-overlapping dominoes that can fit on the board is the total area divided
 * by the area of a single domino. Since we cannot use partial dominoes, we take
 * the floor of this division. In C++, integer division automatically performs
 * this flooring operation.
 *
 * Formula: floor((M * N) / 2)
 *
 * - Time Complexity: O(1), as it involves a single arithmetic calculation.
 * - Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

int main() {
  int s_edge, l_edge; // Represent the M and N dimensions of the board.
  cin >> s_edge >> l_edge;

  // Calculates the maximum number of 2x1 dominoes by dividing the total
  // board area by the area of a single domino (2). Integer division handles
  // the floor operation needed for this tiling problem.
  cout << s_edge * l_edge / 2;
  
  return 0;
}
