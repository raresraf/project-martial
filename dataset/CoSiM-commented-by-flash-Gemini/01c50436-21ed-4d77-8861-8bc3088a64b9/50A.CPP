/**
 * @01c50436-21ed-4d77-8861-8bc3088a64b9/50A.CPP
 * @brief Solution to the "Domino piling" problem from competitive programming.
 *
 * This program efficiently calculates the maximum number of 1x2 or 2x1 dominoes
 * that can be placed on an M x N rectangular board without overlapping.
 * The core insight is that each domino covers exactly two squares, so the maximum
 * number of dominoes is simply half the total area of the board, rounded down.
 *
 * Algorithm: Direct calculation of floor(M * N / 2), leveraging integer division properties.
 * Time Complexity: O(1), as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1), requiring only a few variables for input and calculation.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  /**
   * Functional Utility: Optimizes I/O operations for competitive programming.
   * Disables synchronization with C's stdio and unties cin/cout.
   */
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);
  
  int n, m;
  /**
   * Functional Utility: Reads two integers from standard input, representing
   *                     the dimensions `M` and `N` of the rectangular board.
   * Pre-condition: Input stream contains two space-separated integers.
   * Post-condition: `n` and `m` variables hold the board dimensions.
   */
  scanf("%d %d", &n, &m);
  
  /**
   * Functional Utility: Initializes `i` to calculate the number of dominoes that can
   *                     be placed using only the full pairs of `m` dimension.
   * Calculation: `m / 2` (integer division) gives the number of 2-unit wide strips along `m`.
   *              Multiplying this by `n` effectively calculates `floor(m/2) * n`.
   * Invariant: `i` stores the maximum number of dominoes that can fit into an `n` x `floor(m/2)*2` area.
   */
  int i = m / 2;
  i *= n;
  
  /**
   * Functional Utility: Calculates the dimensions of any remaining 1-unit wide strip
   *                     after the initial `m/2` calculation.
   * `x`: Stores the remainder of `m` divided by 2 (0 or 1), representing the existence
   *      and width of a potential remaining strip along the `m` dimension.
   * `y`: Stores the integer division of `n` by 2, representing how many 2-unit segments
   *      can be formed along the `n` dimension.
   */
  int x = m % 2, y = n / 2;
  /**
   * Block Logic: Conditionally adds dominoes from a remaining 1-unit wide strip of the board.
   * Functional Utility: Accounts for the possibility of placing additional dominoes in a
   *                     `1 x N` or `M x 1` strip that was not fully covered by the initial
   *                     `m/2` calculation.
   * Pre-condition: `x` is the remainder of `m / 2`, and `y` is `n / 2`.
   * Invariant: `i` is updated to include dominoes from this potential strip.
   */
  if (x > 0 && y > 0) { // Check if a 1-unit wide strip exists along `m` (`x > 0`) AND if `n` is large enough to place dominoes (`y > 0`).
    i += x * y; // Functional Utility: Adds `x * y` additional dominoes to the total count `i`.
  }
  
  /**
   * Functional Utility: Prints the final calculated maximum number of dominoes
   *                     that can be placed on the board to standard output,
   *                     followed by a newline character.
   * Post-condition: The result `i` is displayed to the user.
   */
  printf("%d\n", i);
  return 0; // Control Flow: Indicates successful program execution.
}
}
