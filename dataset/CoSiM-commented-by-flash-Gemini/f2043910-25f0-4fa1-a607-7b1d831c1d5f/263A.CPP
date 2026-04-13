/**
 * @file 263A.CPP
 * @brief This program reads a 5x5 integer matrix, identifies the location
 *        of the maximum value within the matrix, and then calculates the
 *        Manhattan distance from that location to the center of the matrix (3,3).
 * Algorithm: Matrix Traversal, Max Element Search, Manhattan Distance.
 * Time Complexity: O(R*C) due to matrix traversal, which is O(1) for a fixed 5x5 matrix.
 * Space Complexity: O(R*C) for the matrix storage and O(1) for auxiliary variables.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Reads integer values into a 2D array from standard input.
 * @param Array The 2D array to populate.
 * @param r The number of rows to read.
 * @param c The number of columns to read.
 * Functional Utility: Populates a given 2D array with user input.
 * Pre-condition: 'Array' is a valid 2D integer array, and 'r' and 'c' are positive integers
 *                representing the dimensions to read.
 * Invariant: After each inner loop iteration, a column in the current row is filled.
 *            After each outer loop iteration, a row is entirely filled.
 */
void read_arr(int Array[10][10], int r, int c) {
  int i, j;
  for (i = 1; i <= r; i++) {
    for (j = 1; j <= c; j++) {
      scanf("%d", &Array[i][j]);
    }
  }
}

// Defines a structure to hold x and y coordinates (1-indexed).
struct location {
  int x;
  int y;
};

int main() {
  int temp = 0; // Variable to store the maximum value found.
  location l;   // Struct to store the 1-indexed coordinates of the maximum value.
  int a, b, c = 5, d, m, n, i, j, Array[10][10], m2[10][10], cnt = 0, r = 5,
            N = 5, M = 5; // Various declared but unused variables.
  // cin>>i,j; // Commented out input, possibly for debugging or an alternative path.
  // Functional Utility: Reads the 5x5 matrix into 'Array'.
  read_arr(Array, r, c);

  /**
   * Block Logic: Traverses the 5x5 matrix to find the maximum value and its 1-indexed coordinates.
   * Invariant: 'temp' holds the maximum value encountered so far, and 'l.x', 'l.y' hold its coordinates.
   */
  for (int i = 1; i <= 5; i++)
    for (int j = 1; j <= 5; j++)
      if (temp < Array[i][j]) {
        temp = Array[i][j];
        l.x = i;
        l.y = j;
      }
  //  cout<<"Maximum Value is "<<temp<<" And is found at
  //  ("<<fabs(l.x-3)+fabs(l.y-3)<<")"; // Commented out output, possibly for debugging.
  // Prints the Manhattan distance from the location of the maximum value to the center (3,3).
  cout << fabs(l.x - 3) + fabs(l.y - 3) << endl;
  // system("pause"); // Commented out system call, likely for Windows-specific pausing in development.
}
