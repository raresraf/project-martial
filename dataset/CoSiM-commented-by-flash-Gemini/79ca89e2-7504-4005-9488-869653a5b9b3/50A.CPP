/**
 * @file 50A.CPP
 * @brief Solution attempt for Codeforces problem 50A - Domino piling.
 *        This problem asks to place the maximum number of 2x1 or 1x2 dominoes
 *        on an M×N rectangular board. Each domino covers two squares.
 * Algorithm (Observed):
 *   1. Reads the dimensions of the board, M and N.
 *   2. Directly calculates and prints (M*N)/2 if either M or N is even. This is the correct
 *      and optimal solution, as each domino covers 2 squares, and the total area is M*N.
 *   3. If both M and N are odd, the code proceeds with a more complex calculation involving
 *      `modedm` and `modedn` (likely M and N rounded down to the nearest even number).
 *      It computes two `outputm` and `outputn` values based on these modified dimensions
 *      and then prints the minimum of these two.
 * Deviation/Correctness:
 *   - The direct calculation `(M*N)/2` is the universally correct solution for this problem
 *     because it simply counts the number of pairs of squares available. The extra logic
 *     for when both M and N are odd is unnecessary and, in this specific implementation,
 *     appears to calculate incorrect values (`outputm` and `outputn` seem to involve
 *     multiplication by 0.5 without proper integer handling for partial dominos, or
 *     a misunderstanding of the problem's constraints).
 * Time Complexity: O(1) as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) for storing a few variables.
 */
#include <iostream>
#include <map>
#include <set>

using namespace std;
int main() {
  /**
   * @brief Main function implementing the solution for the "Domino piling" problem.
   *
   * Reads the dimensions of a rectangular board (n rows, m columns) and calculates
   * the maximum number of 1x2 or 2x1 dominoes that can be placed on it.
   * This implementation provides a partially correct solution, correctly handling
   * cases where one dimension is even, but includes a complex and potentially
   * incorrect calculation for cases where both dimensions are odd.
   * @return int Returns 0 upon successful execution.
   */
  int n;
  int m;
  cin >> n >> m; // Read board dimensions: n (rows), m (columns).
  int outputm;     // Variable to store result of one calculation path.
  int outputn;     // Variable to store result of another calculation path.

  // Block Logic: Handles cases where either M or N is even.
  // Pre-condition: 'm' is the number of columns.
  // Invariant: If 'm' is even, the maximum dominoes is simply half the total area.
  // This is the correct and optimal solution.
  if (m % 2 == 0) {
    cout << (m * n) / 2; // Directly print half the total area.
    return 0; // Terminate early as solution is found.
  } else {
    // Block Logic: Handles the case where 'm' is odd. Further checks if 'n' is even.
    // Pre-condition: 'm' is odd.
    // Invariant: If 'n' is also even, the solution is still half the total area.
    if (n % 2 == 0) {
      cout << (m * n) / 2; // Directly print half the total area.
      return 0; // Terminate early as solution is found.
    }
  }
  // Block Logic: This complex calculation path is reached only when both 'm' and 'n' are odd.
  // It attempts to determine the maximum dominoes by reducing both dimensions to the nearest even number.
  // Note: For an odd x odd board, the maximum number of dominoes is still (M*N)/2.
  // The formulas used here might be an attempt at a different approach or contain logical errors.
  // It effectively calculates two scenarios of tiling and picks the minimum.
  int modedm = m - (m % 2); // 'm' rounded down to the nearest even number.
  int modedn = n - (n % 2); // 'n' rounded down to the nearest even number.
  outputm = 0.5 * (modedm * n) + 0.5 * (modedn); // Calculation for one tiling strategy.
  outputn = 0.5 * (modedn * m) + 0.5 * (modedm); // Calculation for another tiling strategy.

  // Block Logic: Compares the results of the two tiling strategies and prints the larger one.
  // Note: The problem asks for the maximum number of dominoes, so usually `max` would be used.
  // However, given the nature of integer division and the problem, `min` here might be intended
  // to pick a result that fits some implicit constraint. The actual correct answer is `(m*n)/2`.
  if (outputm > outputn) {
    cout << outputn; // Prints the smaller of the two calculated values.
    return 0;        // Terminate early.
  }
  cout << outputm; // Prints the larger of the two calculated values.
  return 0;
}
