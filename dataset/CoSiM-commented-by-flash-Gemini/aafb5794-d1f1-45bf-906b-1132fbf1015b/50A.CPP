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
  int n;
  int n1;
  // Block Logic: Read the dimensions of the rectangular board (N and N1).
  // Pre-condition: User provides two integers representing the board dimensions.
  // Invariant: n and n1 will hold the integer values for board dimensions.
  cin >> n >> n1;

  int s;
  // Block Logic: Calculate the total area of the board.
  // Invariant: 's' will store the product of n and n1.
  s = n * n1;

  int div;
  // Block Logic: Calculate the maximum number of dominoes that can be placed.
  // The total area is 's'. Since each domino covers 2 squares, the maximum number
  // of dominoes is 's' / 2, with integer division correctly handling odd areas.
  // Invariant: 'div' will store the calculated maximum number of dominoes.
  div = s / 2;

  // Block Logic: Print the calculated maximum number of dominoes to standard output.
  // Post-condition: The maximum number of dominoes is printed.
  cout << div;
  return 0;
}
