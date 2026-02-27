/**
 * @file 50A.CPP
 * @brief Solves a tiling problem, likely a variation of "Theatre Square" (1A) from Codeforces.
 * @details The filename "50A.CPP" corresponds to the "Domino piling" problem, but
 * the logic implemented here does not solve that problem. Instead, it calculates
 * the number of 2x2 tiles needed to cover an M x N area, which is characteristic
 * of the "Theatre Square" problem. The implementation contains some unusual
 * edge case handling.
 *
 * @algorithm (as implemented)
 * 1. Read the dimensions of the area, M and N.
 * 2. Define a tile dimension `a` as 2.
 * 3. A special, likely incorrect, condition checks if M or N is <= 2, and if so,
 *    prints N.
 * 4. Otherwise, it calculates the number of tiles needed along each dimension:
 *    - `x = ceil(N / a)`
 *    - `y = ceil(M / a)`
 * 5. It then prints the product `x * y`, which is the total number of tiles.
 *
 * @complexity
 * Time Complexity: O(1), as it involves a constant number of arithmetic operations.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
main() {
  int n, m; // Dimensions of the rectangle to be tiled.
  int a = 2; // Represents the side length of the square tile.
  int x, y; // Number of tiles needed along each dimension.
  cin >> m >> n;

  /**
   * @block
   * @brief This block appears to be an attempt to handle an edge case, but its
   * logic is unclear and likely incorrect for standard tiling problems.
   */
  if (n <= 2 || m <= 2) {
    cout << n << endl;
  } else {
    /**
     * @block
     * @brief Calculates the number of 2x2 tiles to cover the M x N area.
     */
    // Calculate ceil(n / a), the number of tiles for the 'n' dimension.
    if (n % a == 0) {
      x = n / a;
    } else {
      x = n / a + 1;
    }
    // Calculate ceil(m / a), the number of tiles for the 'm' dimension.
    if (m % a == 0) {
      y = m / a;
    } else {
      y = m / a + 1;
    }
    // Total tiles is the product of the tiles needed for each dimension.
    cout << x * y << endl;
  }
}
