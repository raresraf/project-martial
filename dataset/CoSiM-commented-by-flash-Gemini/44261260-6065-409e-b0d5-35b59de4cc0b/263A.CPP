/**
 * @file 263A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 263A - "Beautiful Matrix").
 *
 * This program calculates the minimum number of moves required to bring the
 * single '1' in a 5x5 matrix to its central position. A move consists of swapping
 * two adjacent rows or two adjacent columns. This is equivalent to calculating
 * the Manhattan distance from the '1' to the center of the matrix.
 *
 * This implementation uses 1-based indexing for the matrix, so the center is
 * considered to be at coordinates (3,3).
 *
 * Algorithm: Matrix traversal and Manhattan distance calculation.
 * It iterates through the 5x5 matrix (using 1-based indexing from 1 to 5)
 * to find the coordinates (i, j) of the cell containing the value '1'.
 * Once '1' is found, it calculates the Manhattan distance as `abs(3 - i) + abs(3 - j)`.
 *
 * Time Complexity: O(R*C), where R is the number of rows and C is the number of columns.
 * Since R=5 and C=5, this is effectively O(1) as the number of operations is constant (25 reads).
 * Space Complexity: O(1) as the matrix size is fixed and small, and only a few variables are used.
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++.
using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads a 5x5 matrix, identifies the location of the number '1', and calculates
 * the Manhattan distance to move it to the center (3,3) using 1-based indexing.
 */
int main() {
  int CntI = 0, CntJ = 0; // Variables to store row and column moves (Manhattan distances).
  int data[6][6];         // Declare a 6x6 2D array to support 1-based indexing for a 5x5 matrix.

  // Block Logic: Iterate through a 5x5 grid using 1-based indexing (rows 1-5, columns 1-5).
  // Invariant: 'i' represents the current row index, 'j' represents the current column index.
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> data[i][j]; // Pre-condition: Read an integer for the current cell.
      // Block Logic: Check if the current number is '1'.
      if (data[i][j] == 1) {
        // Functional Utility: Calculate the absolute difference for row moves to the center (row 3).
        if (i < 3)
          CntI = 3 - i;
        else if (i > 3)
          CntI = i - 3;
        // Functional Utility: Calculate the absolute difference for column moves to the center (column 3).
        if (j < 3)
          CntJ = 3 - j;
        else if (j > 3)
          CntJ = j - 3;
      }
    }
  }
  // Output the total Manhattan distance (sum of row and column moves).
  cout << CntI + CntJ << endl; // Use endl for flushing output.
  return 0; // Indicate successful execution.
}
