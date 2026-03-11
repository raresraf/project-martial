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

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to execute the domino piling calculation.
 *
 * Reads the board dimensions, calculates the maximum dominoes, and prints the result.
 * @return 0 on successful execution.
 */
int main() {
  int m, n, s, t, r; // 't' is declared but unused.
  // Block Logic: Read the dimensions of the rectangular board (M and N).
  // Pre-condition: User provides two integers representing the board dimensions.
  // Invariant: m and n will hold the integer values for board dimensions.
  cin >> m >> n;

  // Block Logic: Calculate the total area of the board.
  // Invariant: 's' will store the product of m and n.
  s = m * n;

  // Block Logic: Calculate the maximum number of dominoes that can be placed.
  // The total area is 's'. Since each domino covers 2 squares, the maximum number
  // of dominoes is 's' / 2, with integer division correctly handling odd areas.
  // Invariant: 'r' will store the calculated maximum number of dominoes.
  r = s / 2;

  // Block Logic: Print the calculated maximum number of dominoes to standard output.
  // Post-condition: The maximum number of dominoes is printed, followed by a newline.
  cout << r << endl;
  return 0;
}
