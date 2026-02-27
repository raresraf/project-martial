/**
 * @file 263A.CPP
 * @brief Solution for competitive programming problem: "Beautiful Matrix".
 *
 * @details This program finds the minimum number of moves required to bring the
 *          number '1' to the center of a 5x5 matrix. A move consists of
 *          shifting the '1' one cell up, down, left, or right. The solution is
 *          calculated as the Manhattan distance from the cell containing '1'
 *          to the center cell (2,2) in a 0-based indexed matrix.
 *          Algorithm: Manhattan Distance calculation using adjusted coordinates.
 *          Time Complexity: O(1) as it involves a fixed-size loop (5x5 matrix)
 *                           and constant-time arithmetic operations.
 *          Domain: Competitive Programming.
 */
#include <iostream>
using namespace std;
int main() {
  int a[5][5]; // Declare a 5x5 integer array to represent the matrix.
  // Declare loop counters 'i', 'j', and 'r', 'c' to store the 0-based
  // row and column coordinates of the cell containing '1'.
  int i, j, r, c;
  
  // Block Logic: Iterate through the 5x5 matrix to find the cell containing '1'.
  // Invariant: After this nested loop, 'r' and 'c' will store the 0-based
  //            coordinates of the cell with value '1'.
  for (i = 0; i < 5; i++) { // Outer loop for rows (0 to 4).
    for (j = 0; j < 5; j++) { // Inner loop for columns (0 to 4).
      cin >> a[i][j];       // Read the value of the current cell.
      if (a[i][j] == 1) {   // If the value is '1'.
        r = i;              // Store the row coordinate.
        c = j;              // Store the column coordinate.
      }
    }
  }
  
  // Block Logic: Calculate the absolute difference between the '1''s row
  //              and the center row (2), and store it in 'r'.
  // Precondition: The center row is 2 (0-indexed).
  if (r >= 2)
    r = r - 2; // If '1' is below or at the center, subtract 2.
  else
    r = 2 - r; // If '1' is above the center, subtract from 2.

  // Block Logic: Calculate the absolute difference between the '1''s column
  //              and the center column (2), and store it in 'c'.
  // Precondition: The center column is 2 (0-indexed).
  if (c >= 2)
    c = c - 2; // If '1' is to the right or at the center, subtract 2.
  else
    c = 2 - c; // If '1' is to the left of the center, subtract from 2.
  
  // Output the total number of moves, which is the sum of row and column differences (Manhattan distance).
  cout << r + c << endl;
  return 0; // Indicate successful program termination.
}
