/**
 * @file 50A.CPP
 * @brief Solution for the "Domino piling" problem (Codeforces 50A).
 *
 * @algorithm
 * The problem is to find the maximum number of 2x1 dominoes that can be placed
 * on a rectangular board of size M x N.
 *
 * The solution is a direct mathematical formula:
 * 1. The total area of the board is M * N.
 * 2. The area of a single domino is 2 * 1 = 2.
 * 3. The maximum number of dominoes is the total board area divided by the
 *    area of one domino.
 * 4. Using integer division, `(M * N) / 2` gives the floor of this value, which
 *    is the correct answer as we can only place whole dominoes.
 *
 * @complexity
 * - Time Complexity: O(1), as it involves a few basic arithmetic operations.
 * - Space Complexity: O(1), as only a few variables are used.
 */
#include <cstdio>
// Note: This solution uses C-style I/O (<cstdio>) rather than C++ streams (<iostream>).
using namespace std;

int main() {
  int M = 0, N = 0; // M and N are the dimensions of the rectangular board.
  // Read the board dimensions using C-style formatted input. The `\n` is not necessary.
  scanf("%d %d\n", &M, &N);
  /**
   * @brief Calculate and print the maximum number of dominoes.
   * The total area is M * N. Each domino covers 2 squares.
   * Integer division `(M * N) / 2` correctly computes floor(area / 2).
   */
  printf("%d\n", M * N / 2);
  return 0;
}
