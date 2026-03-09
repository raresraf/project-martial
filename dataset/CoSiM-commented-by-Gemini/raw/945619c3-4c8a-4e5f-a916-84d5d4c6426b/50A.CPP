/**
 * @file 50A.CPP
 * @brief Calculates the maximum number of dominoes on a rectangular board.
 *
 * This program solves the "Domino piling" problem from Codeforces. Given a
 * rectangular board of size M x N, it calculates the maximum number of 2x1
 * dominoes that can be placed on it without overlapping.
 *
 * @algorithm Mathematical Formula
 * @complexity
 * - Time: O(1), as it performs a single arithmetic calculation.
 * - Space: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  // n and m are the dimensions of the rectangular board.
  int n, m;
  cin >> n >> m;

  /**
   * @brief Functional Utility: Calculates the maximum number of dominoes.
   * Logic: The total area of the board is M * N. Each domino has an area of 2.
   * The maximum number of dominoes is the total board area divided by the
   * area of a single domino. Integer division `(m * n) / 2` correctly
   * gives the floor of this value, which is the answer.
   */
  cout << m * n / 2 << endl;
  
  return 0;
}
