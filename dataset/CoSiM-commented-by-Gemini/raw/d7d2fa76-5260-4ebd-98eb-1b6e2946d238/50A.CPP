/**
 * @file 50A.CPP
 * @brief Solution for Codeforces Problem 50A - "Domino piling".
 * @details The problem is to find the maximum number of 2x1 dominoes that can be placed on a rectangular
 * board of size M x N.
 *
 * @author (unknown)
 *
 * @algorithm
 * The total area of the board is M * N.
 * Each domino covers an area of 2.
 * Therefore, the maximum number of dominoes that can fit on the board is the total area
 * divided by the area of a single domino.
 * The formula is `floor((M * N) / 2)`. In C++, integer division `(m * n) / 2`
 * automatically performs the floor operation.
 *
 * @complexity
 * Time Complexity: O(1), as it only involves a few arithmetic operations.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

int main() {
  int m, n;
  // Read the dimensions of the board.
  cin >> m >> n;

  /**
   * @brief Calculates and prints the maximum number of dominoes.
   * The total area is m * n. Each domino has an area of 2.
   * Integer division `(m * n) / 2` gives the maximum number of non-overlapping
   * dominoes that can fit.
   */
  cout << (m * n) / (2) << endl;
}