/**
 * @file 50A.CPP
 * @brief Solution to the "Domino piling" problem.
 *
 * This program calculates the maximum number of 2x1 dominoes that can be placed
 * on a M x N rectangular board.
 * Algorithm: The problem can be solved by observing that each domino covers two
 * squares. Therefore, the maximum number of dominoes is half the total area
 * of the board, rounded down.
 * Time Complexity: O(1) - constant time operations for input, calculation, and output.
 * Space Complexity: O(1) - constant space used for variables.
 */

#include <iostream>
using namespace std;

// Global variables for board dimensions (m, n) and result (a).
int m, n, a;

/**
 * @brief Main function to execute the domino piling calculation.
 *
 * Reads the board dimensions, calculates the maximum dominoes, and prints the result.
 * @return 0 on successful execution.
 */
int main() {
  // Block Logic: Read the dimensions of the rectangular board (m and n).
  // Pre-condition: User provides two integers representing the board dimensions.
  // Invariant: m and n will hold the integer values for board dimensions.
  cin >> m >> n;

  // Block Logic: Calculate the maximum number of 2x1 dominoes that can be placed.
  // If the total area (m * n) is even, half the area can be covered.
  // If the total area is odd, one square will be left uncovered, so (area - 1) / 2 dominoes can be placed.
  // This effectively performs integer division of (m * n) / 2, rounding down.
  // Invariant: 'a' will store the calculated maximum number of dominoes.
  if ((m * n) % 2 == 0) {
    a = (m * n) / 2;
  } else {
    a = ((m * n) - 1) / 2;
  }

  // Block Logic: Print the calculated maximum number of dominoes to standard output.
  // Post-condition: The maximum number of dominoes is printed.
  cout << a;
}