/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem from Codeforces using C-style I/O.
 * @details The problem is to find the maximum number of 2x1 dominoes that can
 * be placed on a rectangular board of size M x N.
 *
 * @algorithm
 * The solution is a direct application of an area formula.
 * 1. The total area of the board is M * N.
 * 2. Each domino has an area of 2.
 * 3. The maximum number of non-overlapping dominoes is the total area divided
 *    by the area of a domino, rounded down.
 * 4. C/C++ integer division `(M * N) / 2` naturally computes this `floor` value.
 *
 * @complexity
 * Time Complexity: O(1). The solution is a single arithmetic calculation.
 * Space Complexity: O(1).
 */
#include <stdio.h>
int main() {
  int x, y; // The dimensions of the rectangular board.
  int d;    // The result: maximum number of dominoes.
  
  scanf("%d %d", &x, &y);
  
  // The maximum number of dominoes is the total area divided by the area of one domino (2).
  // Integer division handles the floor operation automatically.
  d = x * y / 2;
  
  printf("%d\n", d);
  return 0;
}
