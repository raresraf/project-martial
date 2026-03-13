/**
 * @file 50A.CPP
 * @brief Solution for CodeForces Problem 50A - "Domino piling".
 *
 * @details
 * The problem is to find the maximum number of 2x1 dominoes that can be placed on
 * a rectangular board of size M x N.
 *
 * @algorithm
 * The total area of the board is M * N. Each domino covers an area of 2.
 * Therefore, the maximum number of dominoes that can fit is the total area
 * divided by the area of a single domino. Since we can't use half a domino,
 * we use integer division.
 *
 * The direct formula is simply `(m * n) / 2`.
 *
 * The code implements this with slightly more complex case analysis:
 * 1. If M is even, it calculates the result as `(m / 2) * n`. This correctly
 *    covers the board.
 * 2. If M is odd, it handles the first `m-1` rows as `(m / 2) * n`, and then for the
 *    last single row of size 1xN, it adds `n / 2` dominoes.
 *
 * @time_complexity O(1), as it's a direct calculation.
 * @space_complexity O(1).
 */
#include <iostream>
using namespace std;

int main() {
  int m, n; // Dimensions of the board.
  cin >> m >> n;
  
  /**
   * @brief Direct calculation for the maximum number of dominoes.
   * The total number of 1x1 squares is `m * n`. Each domino covers two squares.
   * Integer division gives the maximum number of non-overlapping dominoes.
   */
  int count = (m * n) / 2;
  
  cout << count << endl;
  
  return 0;
}
