/**
 * @file 50A.CPP
 * @brief Optimal solution for the "Domino Piling" problem.
 *
 * @problem Given a rectangular board of size M x N, find the maximum number
 * of 2x1 dominoes that can be placed on it.
 *
 * @algorithm The problem has a direct mathematical solution. The total area of the
 * board is M * N. Each domino covers an area of 2. Therefore, the maximum
 * number of dominoes that can fit is the total area divided by 2, using
 * integer division to automatically handle boards with an odd-numbered area.
 * The code implements this with a single, efficient calculation.
 *
 * @time_complexity O(1), as it involves a constant number of operations.
 * @space_complexity O(1).
 */
#include <iostream>

using namespace std;

int main() {
  int m, n;
  // Read the dimensions of the board.
  cin >> m >> n;

  // Calculate and print the maximum number of dominoes.
  // Integer division `(m * n) / 2` correctly handles both even and odd areas.
  cout << (m * n) / 2 << endl;

  return 0;
}
