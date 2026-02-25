/**
 * @file 50A.CPP
 * @brief Solution for Codeforces Problem 50A - Domino piling.
 *
 * This program calculates the maximum number of 2x1 dominoes that can be
 * placed on a rectangular M x N board. Each domino covers exactly two squares.
 *
 * Algorithm: Simple area division. Since each domino covers two squares,
 * the maximum number of dominoes that can be placed on a board is half
 * of the total area of the board (M * N). Integer division in C++
 * automatically handles the floor operation, which is appropriate here.
 * Time Complexity: O(1) - constant time operations.
 * Space Complexity: O(1) - constant space usage.
 */

#include <iostream> // Required for standard input/output operations (cin, cout).

using namespace std; // Uses the standard namespace.

int main() {
  /**
   * @brief Main function to execute the Domino Piling problem logic.
   *
   * Reads the dimensions M and N of the rectangular board from standard input.
   * It calculates the total area of the board and then determines the maximum
   * number of 2x1 dominoes that can fit by dividing the total area by 2.
   * Finally, it prints this result.
   */
  // Declare integer variables for board dimensions 'm', 'n', and 'area'.
  int m, n, area;
  // Define the area covered by one domino (2 squares).
  int areaOneDominoes = 2;

  // Read the dimensions of the board (M and N).
  cin >> m >> n;
  // Calculate the total area of the rectangular board.
  area = m * n;
  // Calculate and print the maximum number of dominoes. Integer division
  // automatically handles floor(), which is what's needed.
  cout << area / areaOneDominoes << endl;

  return 0; // Indicate successful program execution.
}