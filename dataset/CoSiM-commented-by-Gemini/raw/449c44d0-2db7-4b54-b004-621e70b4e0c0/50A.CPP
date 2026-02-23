/**
 * @file 50A.CPP
 * @brief Solution to Codeforces problem 50A - Domino piling.
 *
 * @algorithm
 * The problem asks for the maximum number of 2x1 dominoes that can be placed
 * on a rectangular board of size M x N.
 *
 * The total area of the board is M * N. Each domino has an area of 2.
 * The maximum number of non-overlapping dominoes that can fit on the board is
 * the total area of the board divided by the area of a single domino.
 *
 * The formula is `floor((M * N) / 2)`.
 *
 * In C++, integer division `(m * n) / 2` automatically performs the floor
 * operation, providing the correct result.
 *
 * @complexity
 * - Time Complexity: O(1), as the solution involves a few arithmetic operations.
 * - Space Complexity: O(1), as only a few variables are used.
 */
#include "stdio.h"

int main() {
  /* n, m: the dimensions of the rectangular board. */
  int n, m;
  scanf("%d%d", &n, &m);

  /*
   * The maximum number of dominoes is the total area of the board (n * m)
   * divided by the area of a single domino (2). Integer division handles
   * the flooring of the result.
   */
  printf("%d", (n * m) / 2);

  return 0;
}
