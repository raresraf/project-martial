/**
 * @file 50A.CPP
 * @brief This program calculates the maximum number of 1x2 or 2x1 dominoes that can be
 *        placed on a rectangular board of size M x N without overlapping.
 *
 * Algorithm:
 * 1. Read the dimensions of the board, M and N.
 * 2. Initialize a counter 'num' to 0.
 * 3. Iteratively place dominoes along the 'M' dimension:
 *    - While 'M' is 2 or more, add 'N' (representing one full row/column of dominoes) to 'num'
 *      and reduce 'M' by 2.
 * 4. If a 1xN strip remains (M becomes 1), add N/2 dominoes (integer division) to 'num'.
 * 5. Print the final count 'num'.
 *
 * Time Complexity: O(M) in the worst case (if M is large), due to the while loop.
 *                  However, since M is typically small (up to 16), it's practically O(1).
 * Space Complexity: O(1) as only a few integer variables are used.
 */

#include <iostream>

using namespace std;
/**
 * @brief Entry point of the program.
 * Functional Utility: Orchestrates reading board dimensions, calculating
 * the maximum dominoes by iteratively covering the board, and printing the result.
 * @return 0 on successful execution.
 */
int main() {
  // Block Logic: Declares integers 'm' and 'n' for board dimensions,
  //              and 'num' to store the calculated maximum number of dominoes, initialized to 0.
  int m, n, num = 0;
  // Block Logic: Reads the two integer dimensions (m and n) of the board from standard input.
  cin >> m >> n;
  /**
   * Block Logic: Iteratively places dominoes in 2xN strips (or Nx2, conceptually).
   * For every 2 units of the 'm' dimension, 'n' dominoes can be placed along that strip.
   *
   * Pre-condition: 'm' and 'n' represent valid positive board dimensions.
   * Invariant: 'num' accumulates the count of dominoes placed in 2-unit wide strips.
   * 'm' is reduced by 2 in each iteration until it is less than 2.
   */
  while (m >= 2) {
    // Inline: Adds 'n' dominoes (representing one full strip) to the total count.
    num = num + n;
    // Inline: Reduces the 'm' dimension by 2, indicating that a 2xN strip has been covered.
    m = m - 2;
  }
  /**
   * Block Logic: Handles the remaining portion of the board if 'm' is 1 (a 1xN strip).
   * In a 1xN strip, N/2 dominoes can be placed (integer division).
   *
   * Pre-condition: 'm' is either 0 or 1 after the while loop.
   * Invariant: If 'm' is 1, 'num' is updated with additional dominoes from the remaining strip.
   */
  if (m == 1)
    // Inline: Adds N/2 dominoes (rounded down) for the remaining 1xN strip.
    num = num + (n / 2);
  // Block Logic: Prints the total calculated maximum number of dominoes, followed by a newline.
  cout << num << endl;
  return 0;
}
