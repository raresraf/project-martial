/**
 * @file 50A.CPP
 * @brief Solution to the "Domino piling" problem.
 *
 * This program calculates the maximum number of 2x1 dominoes that can be placed
 * on a M x N rectangular board.
 * Algorithm: The problem can be solved by observing that each domino covers two
 * squares. Therefore, the maximum number of dominoes is half the total area
 * of the board, rounded down. The logic specifically checks for even dimensions
 * to simplify the calculation, otherwise handles the odd total area case.
 * Time Complexity: O(1) - constant time operations for input, calculation, and output.
 * Space Complexity: O(1) - constant space used for variables.
 */

#include <iostream> // Required for standard input/output operations (std::cin, std::cout).
using namespace std; // Uses the standard namespace for convenience.

/**
 * @brief Main function to execute the domino piling calculation.
 *
 * Reads the board dimensions, calculates the maximum dominoes, and prints the result.
 * @return 0 on successful execution.
 */
int main() {
  int m, n; // Declare variables for board dimensions.

  // Block Logic: Read the dimensions of the rectangular board (m and n).
  // Pre-condition: User provides two integers representing the board dimensions.
  // Invariant: m and n will hold the integer values for board dimensions.
  cin >> m >> n;

  // Block Logic: Calculate and print the maximum number of 2x1 dominoes.
  // If either 'm' or 'n' is even, the total area (m * n) will be even,
  // allowing an exact division by 2 to get the domino count.
  // If both 'm' and 'n' are odd, the total area (m * n) will be odd.
  // In this case, one square will be left uncovered, so ((m * n) - 1) / 2 dominoes can be placed.
  // This approach correctly handles integer division for both even and odd total areas.
  // Post-condition: The maximum number of dominoes is printed to standard output.
  if (m % 2 == 0 || n % 2 == 0) {
    cout << (m * n / 2);
  } else {
    cout << ((m * n) - 1) / 2;
  }

  return 0; // Indicate successful execution.
}