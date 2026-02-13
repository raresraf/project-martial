/**
 * @file 50A.CPP
 * @brief Solution for the "Domino piling" problem (Codeforces 50A).
 * @author competitive-programmer
 *
 * @details This program calculates the maximum number of 2x1 dominoes that can
 * be placed on a rectangular board of size M x N without overlapping.
 *
 * Algorithm:
 * 1. The total area of the board is M * N.
 * 2. The area of a single domino is 2 * 1 = 2.
 * 3. The maximum number of dominoes is the total area of the board divided by
 *    the area of one domino. Integer division `(M * N) / 2` naturally gives
 *    the correct answer, as any remaining single square cannot fit a domino.
 *
 * Time Complexity: O(1), as it's a single arithmetic calculation.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

int main() {
  // M and N are the dimensions of the rectangular board.
  int M, N;
  cin >> M >> N;

  /**
   * The maximum number of 2x1 dominoes that can fit on the board is calculated
   * by finding the total area (M * N) and performing an integer division by
   * the area of a single domino (2).
   */
  cout << M * N / 2 << endl;

  return 0;
}
