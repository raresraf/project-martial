/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem from Codeforces.
 *
 * This program calculates the maximum number of 2x1 dominoes that can be
 * placed on a rectangular board of size M x N.
 *
 * @algorithm The total area of the board is M * N. Each domino covers an area
 * of 2. Therefore, the maximum number of dominoes that can fit is the total
 * board area divided by 2. The code implements this using integer division.
 *
 * @time_complexity O(1).
 * @space_complexity O(1).
 */
#include <iostream>
using namespace std;
int m, n, a;
int main() {
  cin >> m >> n;
  /**
   * @block_logic Calculates the maximum number of dominoes. This is equivalent
   * to the integer division `(m * n) / 2`.
   */
  if ((m * n) % 2 == 0) {
    a = (m * n) / 2;
  } else {
    a = ((m * n) - 1) / 2;
  }
  cout << a;
}