/**
 * @file 50A.CPP
 * @brief Solution for Codeforces Problem 50A - "Domino piling".
 *
 * This program calculates the maximum number of 2x1 dominoes that can be placed
 * on a rectangular board of size M x N. The formula is derived from the total
 * area of the board divided by the area of a single domino.
 *
 * @algorithm Direct calculation.
 * @time_complexity O(1), as it performs a constant number of arithmetic
 *                  operations.
 * @space_complexity O(1), as it only stores two integers.
 */
#include <iostream>

using namespace std;

/**
 * @brief Main function to calculate the number of dominoes.
 */
int main() {
  int m, n;

  // Read the dimensions of the board.
  cin >> m >> n;

  /**
   * @brief Calculate and print the maximum number of dominoes.
   * The total area of the board is M * N. Each domino covers an area of 2.
   * Therefore, the maximum number of dominoes that can fit is the total
   * area divided by 2, using integer division.
   */
  cout << (m * n) / 2 << endl;
}
