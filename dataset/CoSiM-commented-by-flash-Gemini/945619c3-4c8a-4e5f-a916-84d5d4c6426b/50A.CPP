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

/**
 * @brief Main function to execute the domino piling calculation.
 *
 * Reads the board dimensions, calculates the maximum dominoes, and prints the result.
 * @return 0 on successful execution.
 */
int main() {
  int n, m;
  // Block Logic: Read the dimensions of the rectangular board (N and M).
  // Pre-condition: User provides two integers representing the board dimensions.
  // Invariant: n and m will hold the integer values for board dimensions.
  cin >> n >> m;

  // Block Logic: Calculate and print the maximum number of dominoes that can be placed.
  // The total area is N * M. Since each domino covers 2 squares, the maximum number
  // of dominoes is (N * M) / 2, with integer division correctly handling odd areas.
  // Post-condition: The maximum number of dominoes is printed to standard output.
  cout << m * n / 2 << endl;
  return 0;
}