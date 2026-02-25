/**
 * @file 263A.CPP
 * @brief Solution for Codeforces Problem 263A - Beautiful Matrix.
 *
 * This program calculates the minimum number of moves required to bring the
 * integer '1' to the center cell (row 3, column 3) of a 5x5 matrix.
 * A move consists of swapping an element with an adjacent one (up, down, left, or right).
 *
 * Algorithm: Manhattan Distance. The minimum number of swaps required is
 * equivalent to the Manhattan distance between the current position of '1'
 * and the target center position (3, 3).
 * Time Complexity: O(R*C), where R and C are the dimensions of the matrix (5x5),
 *                  for reading input. The calculation itself is O(1).
 * Space Complexity: O(R*C) for storing the matrix.
 */

#include <algorithm> // Required for std::abs.
#include <bitset>    // Not used in this specific solution.
#include <cassert>   // Not used.
#include <cmath>     // Not strictly needed for abs with integers, but often included.
#include <cstdio>    // Not used in this specific solution (uses iostream).
#include <cstdlib>   // Not used.
#include <cstring>   // Not used.
#include <ctime>     // Not used.
#include <deque>     // Not used.
#include <functional> // Not used.
#include <iomanip>   // Not used.
#include <iostream>  // Required for standard input/output (cin, cout).
#include <list>      // Not used.
#include <map>       // Not used.
#include <memory.h>  // Not used.
#include <numeric>   // Not used.
#include <set>       // Not used.
#include <sstream>   // Not used.
#include <stack>     // Not used.
#include <utility>   // Not used.
#include <vector>    // Not used.

using namespace std; // Uses the standard namespace.

// Declare a global 5x5 integer array to store the matrix.
// Using 1-based indexing for convenience to match problem statement (rows/cols 1-5).
int arr[5][5];

int main(int argc, char const *argv[]) {
  /**
   * @brief Main function to execute the Beautiful Matrix problem logic.
   *
   * Reads a 5x5 matrix from standard input. It then finds the row and
   * column coordinates of the element '1' within this matrix.
   * Finally, it calculates and prints the Manhattan distance from this
   * position to the center of the matrix (3, 3).
   */
  // Declare 'x' and 'y' to store the row and column coordinates of the '1'.
  int x, y;
  // Block Logic: Read the 5x5 matrix and find the position of '1'.
  // Invariant: The nested loops iterate through all cells of the 5x5 matrix.
  // Pre-condition: The matrix is guaranteed to contain exactly one '1'.
  for (int i = 1; i <= 5; i++) { // Iterate through rows (1 to 5).
    for (int j = 1; j <= 5; j++) { // Iterate through columns (1 to 5).
      cin >> arr[i][j]; // Read the value for the current cell.
      // If the current cell contains '1', store its coordinates.
      if (arr[i][j] == 1) {
        x = i; // Store row index.
        y = j; // Store column index.
      }
    }
  }

  // Calculate the Manhattan distance to move '1' to the center (3,3).
  // Manhattan distance = |x1 - x2| + |y1 - y2|.
  int test = abs(3 - x) + abs(3 - y);
  // Output the calculated minimum number of moves.
  cout << test;
  return 0; // Indicate successful program execution.
}