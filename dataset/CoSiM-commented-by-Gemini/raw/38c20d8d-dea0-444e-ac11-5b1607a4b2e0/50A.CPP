/**
 * @file 50A.CPP
 * @brief Calculates the maximum number of dominoes on a rectangular board.
 *
 * This program solves the classic tiling problem of finding the maximum number
 * of 2x1 dominoes that can be placed on a rectangular board of size M x N
 * without overlapping.
 *
 * Algorithm:
 * 1. Read the dimensions of the board, M and N.
 * 2. Calculate the total area of the board, which is M * N.
 * 3. Each domino has an area of 2 (since it is 2x1).
 * 4. The maximum number of non-overlapping dominoes is the total area of the
 *    board divided by the area of a single domino. Integer division naturally
 *    handles cases where one square might be left over.
 * 5. The result is `(M * N) / 2`.
 * 6. Print the calculated maximum number of dominoes.
 *
 * Time Complexity: O(1), as the solution is a direct arithmetic calculation.
 * Space Complexity: O(1), as it uses only a few variables to store the dimensions.
 */
#include <cstdio>
using namespace std;

int main() {
  int n, m;
  // Read the dimensions of the rectangular board.
  scanf("%d%d", &m, &n);
  
  /**
   * @brief Calculate and print the maximum number of dominoes.
   * The total area is m * n. Each domino covers 2 squares.
   * Integer division `(m * n) / 2` gives the maximum number of
   * dominoes that can fit.
   */
  printf("%d", (n * m) / 2);
}
