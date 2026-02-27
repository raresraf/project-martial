/**
 * @file 263A.CPP
 * @brief Solution for competitive programming problem: "Beautiful Matrix".
 *
 * @details This program finds the minimum number of moves required to bring the
 *          number '1' to the center of a 5x5 matrix. A move consists of
 *          shifting the '1' one cell up, down, left, or right. The solution is
 *          calculated as the Manhattan distance from the cell containing '1'
 *          to the center cell (3,3) in a 1-based indexed matrix.
 *          Algorithm: Manhattan Distance calculation.
 *          Time Complexity: O(1) as it involves a fixed-size loop (5x5 matrix)
 *                           and constant-time arithmetic operations.
 *          Domain: Competitive Programming.
 */
// problem : https://codeforces.com/problemset/problem/263/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  // Declare variables:
  // 'a': stores the value of the current cell read from input.
  // 'r', 'c': store the 1-based row and column coordinates of the cell containing '1'.
  int a, r, c;
  // Loop counters.
  int i, j;
  
  // Block Logic: Iterate through a 5x5 matrix to find the cell containing '1'.
  // Invariant: 'r' and 'c' will store the 1-based coordinates of '1' after the loops.
  for (i = 1; i <= 5; i++) { // Outer loop for rows (1 to 5).
    for (j = 1; j <= 5; j++) { // Inner loop for columns (1 to 5).
      scanf("%d", &a); // Read the value of the current cell.
      if (a) {         // If 'a' is non-zero (i.e., it's '1').
        r = i;         // Store the row coordinate.
        c = j;         // Store the column coordinate.
      }
    }
  }

  // Calculate the Manhattan distance from the found '1' (r,c) to the center (3,3).
  // The Manhattan distance is |x1 - x2| + |y1 - y2|.
  printf("%d", abs(3 - r) + abs(3 - c));
}