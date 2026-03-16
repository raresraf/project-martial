/**
 * @file 50A.CPP
 * @brief Solution to the "Domino piling" problem (Codeforces 50A).
 *
 * @details
 * This program calculates the maximum number of 2x1 dominoes that can be placed on a
 * rectangular board of size M x N.
 *
 * The code uses an iterative approach to simulate the tiling process. It repeatedly
 * covers 2xN sections of the board, adding `n` dominoes for each section. If an
 * odd row remains at the end, it tiles that 1xN section with `n/2` dominoes.
 *
 * While this iterative logic is correct, a more direct mathematical solution exists.
 * The total area is M*N, and each domino covers an area of 2. Therefore, the maximum
 * number of dominoes is simply `floor((M * N) / 2)`, which can be calculated in C++
 * using integer division: `(m * n) / 2`.
 *
 * Algorithm (as implemented): Iterative Tiling Simulation
 * Time Complexity: O(M)
 * Space Complexity: O(1)
 */
#include <iostream>

using namespace std;
int main() {
  int m, n, num = 0;
  cin >> m >> n;

  /**
   * @brief This loop tiles the majority of the board in 2xN chunks.
   * @details For every two rows of the board, `n` dominoes can be placed horizontally.
   * This loop continues until fewer than two rows remain.
   */
  while (m >= 2) {
    num = num + n;
    m = m - 2;
  }

  /**
   * @brief Handles the case where M is odd, leaving one final row.
   * @details If a single 1xN row remains, `floor(n/2)` dominoes can be placed
   * vertically within it.
   */
  if (m == 1)
    num = num + (n / 2);
  cout << num << endl;
  return 0;
}