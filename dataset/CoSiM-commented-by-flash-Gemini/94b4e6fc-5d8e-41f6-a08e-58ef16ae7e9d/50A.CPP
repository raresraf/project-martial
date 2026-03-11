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
  int M;
  int N;
  // Block Logic: Read the dimensions of the rectangular board (M and N).
  // Pre-condition: User provides two integers representing the board dimensions.
  // Invariant: M and N will hold the integer values for board dimensions.
  cin >> M;
  cin >> N;

  int area = M * N;
  // Block Logic: Adjust the area for odd-sized boards to ensure correct integer division for dominoes.
  // If the total area is odd, one square cannot be covered by a domino, so the effective area
  // for domino placement is reduced by one.
  // Invariant: 'area' will be an even number if it was initially odd, otherwise unchanged.
  if (area % 2 != 0) {
    area--;
  }

  // Block Logic: Calculate and print the maximum number of dominoes that can be placed.
  // Each domino covers 2 squares, so dividing the effective area by 2 gives the maximum count.
  // Post-condition: The maximum number of dominoes is printed to standard output.
  cout << area / 2;
  return 0;
}