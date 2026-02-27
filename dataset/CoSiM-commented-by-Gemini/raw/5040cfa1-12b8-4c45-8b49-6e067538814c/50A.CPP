/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem from Codeforces.
 * @details The problem is to find the maximum number of 2x1 dominoes that can
 * be placed on a rectangular board of size M x N without overlapping.
 *
 * @algorithm
 * 1. Read the dimensions of the board, M and N.
 * 2. Calculate the total area of the board, which is M * N.
 * 3. Each domino covers an area of 2. The maximum number of dominoes that can
 *    fit is the total area of the board divided by the area of one domino.
 * 4. This is equivalent to floor((M * N) / 2), which in integer arithmetic is
 *    simply (M * N) / 2.
 * 5. The code implements this by checking if the total area is even or odd, but
 *    the result is the same as integer division.
 * 6. Print the result.
 *
 * @complexity
 * Time Complexity: O(1), as the solution involves a few arithmetic operations.
 * Space Complexity: O(1), as only a few variables are used.
 */
#include <stdio.h>

int main(void) {
  int m; // Represents the first dimension of the board.
  int n; // Represents the second dimension of the board.

  scanf("%d %d", &m, &n);

  /**
   * @brief Calculate the maximum number of dominoes.
   * @details The logic calculates floor((m * n) / 2).
   * If the total area (m * n) is even, the number of dominoes is exactly half the area.
   * If the area is odd, one square will be left over, so the number of dominoes
   * is (area - 1) / 2.
   * Both cases are handled by simple integer division: (m * n) / 2.
   */
  if (m * n % 2 == 0) {
    printf("%d", m * n / 2);
  } else {
    printf("%d", (m * n - 1) / 2);
  }

  return 0;
}
