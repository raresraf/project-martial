/**
 * @file 50A.CPP
 * @brief Solution for a grid-tiling problem, likely "Domino Piling" or "Chessboard and Dominoes".
 *        This program calculates the maximum number of 2x2 tiles that can be placed
 *        on an M x N grid, with a special case for smaller grids.
 *        The interpretation here is that each "tile" effectively covers a 2x2 area,
 *        and the goal is to find how many such tiles can fit.
 *
 * Algorithm: Integer division with ceiling for both dimensions and multiplication.
 *            For an M x N grid, it calculates how many 2-unit segments fit along
 *            each dimension (rounding up) and then multiplies these counts.
 *            A special condition handles cases where one or both dimensions are small.
 * Time Complexity: O(1) as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) as it uses a fixed amount of memory regardless of input size.
 */

#include <bits/stdc++.h> // A common competitive programming header, includes various standard libraries.
using namespace std;

// The `main` function without an explicit return type defaults to `int` in C++.
// In competitive programming, `main()` is sometimes used for brevity.
main() {
  /**
   * Functional Utility: Reads grid dimensions M and N, then calculates and prints
   *                     the maximum number of 2x2 tiles that can fit into the grid.
   */
  int n, m, a = 2, x, y; // n: width, m: height, a: tile dimension (2), x, y: calculated tile counts per dimension.
  cin >> m >> n; // Reads the grid dimensions, height (m) and width (n).

  /**
   * Block Logic: Handles edge cases where one or both dimensions are too small (<= 2)
   *              for standard 2x2 tiling.
   * Pre-condition: `m` and `n` are initialized.
   * Invariant: For small grids, `n` is printed; otherwise, the tiling calculation proceeds.
   *            The interpretation of outputting `n` for small grids implies a different
   *            tiling rule or problem variant for these edge cases.
   */
  if (n <= 2 || m <= 2) {
    cout << n << endl; // Special case output.
  } else {
    /**
     * Block Logic: Calculates the number of 2x2 tiles that can fit along the width (n).
     *              Uses ceiling division to account for partial fits.
     * Pre-condition: `n` is greater than 2.
     * Invariant: `x` will hold the number of 2-unit segments that fit into `n`.
     */
    if (n % a == 0) { // Condition Check: If `n` is perfectly divisible by `a` (2).
      x = n / a;      // Direct division.
    } else {
      x = n / a + 1; // Ceiling division for odd `n`.
    }

    /**
     * Block Logic: Calculates the number of 2x2 tiles that can fit along the height (m).
     *              Uses ceiling division to account for partial fits.
     * Pre-condition: `m` is greater than 2.
     * Invariant: `y` will hold the number of 2-unit segments that fit into `m`.
     */
    if (m % a == 0) { // Condition Check: If `m` is perfectly divisible by `a` (2).
      y = m / a;      // Direct division.
    } else {
      y = m / a + 1; // Ceiling division for odd `m`.
    }
    cout << x * y << endl; // Prints the total number of 2x2 tiles that can fit.
  }
}
