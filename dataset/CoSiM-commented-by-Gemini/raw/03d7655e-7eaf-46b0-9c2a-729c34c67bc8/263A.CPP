/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem from Codeforces contest 263.
 *
 * @details
 * The problem requires finding the minimum number of moves to make a 5x5 matrix "beautiful".
 * A matrix is considered beautiful if the single digit '1' it contains is located at the center,
 * which corresponds to the intersection of the 3rd row and 3rd column. A move is defined as
 * swapping any two adjacent rows or any two adjacent columns.
 *
 * The solution approach is to calculate the Manhattan distance between the position of the '1'
 * and the center of the matrix (3,3). The Manhattan distance |x1 - x2| + |y1 - y2| directly
 * corresponds to the minimum number of single row or column swaps required to move an element
 * from (x1, y1) to (x2, y2) in a grid.
 *
 * Algorithm:
 * 1. Read the 5x5 matrix from standard input.
 * 2. As the matrix is read, identify the coordinates (row and column) of the element '1'.
 * 3. Calculate the absolute difference between the row of '1' and the target row (3).
 * 4. Calculate the absolute difference between the column of '1' and the target column (3).
 * 5. The sum of these two differences gives the minimum number of moves.
 *
 * Time Complexity: O(1), as the matrix size is fixed at 5x5, the loops run a constant number of times.
 * Space Complexity: O(1), for storing the 5x5 matrix.
 */

#include <bits/stdc++.h> // Includes all standard C++ libraries
using namespace std;

// Macros for cleaner and more concise code
#define in cin >> // Macro for input
#define out cout << // Macro for output
#define FORI(i, k, l) for (int i(k); i < l; i++) // Loop macro for rows
#define FORJ(i, k, l) for (int j(k); j < l; j++) // Loop macro for columns
#define Vi vector<int> // Type alias for vector of integers
#define Msi map<string, int> // Type alias for map of string to int
#define Mii map<int, int> // Type alias for map of int to int

int main(void) {
  int arr[5][5]; // 2D array to store the 5x5 matrix
  int x = 0, y = 0; // Variables to store the coordinates of the '1'

  // Block Logic: Iterates through each element of the 5x5 grid to find the '1'.
  // Pre-condition: The matrix contains exactly one '1'.
  // Invariant: After each iteration of the inner loop, all elements up to arr[i][j] have been checked.
  FORI(i, 0, 5) {
    FORJ(j, 0, 5) {
      in(arr[i][j]);
      if (arr[i][j] == 1) {
        // Store the 1-indexed coordinates of '1' for calculating the distance.
        x = j + 1; // 1-indexed column
        y = i + 1; // 1-indexed row
      }
    }
  }

  // Functional Utility: Calculates the Manhattan distance from the current position of '1'
  // to the center of the matrix (3, 3). This distance is equivalent to the minimum number of moves.
  x = abs(x - 3);
  y = abs(y - 3);

  // Output the result.
  out(x + y);
}
