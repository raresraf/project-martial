/**
 * @file 263A.CPP
 * @brief Solution for the "Beautiful Matrix" problem (Codeforces 263A).
 * @author competitive-programmer
 *
 * @details The problem involves a 5x5 matrix containing a single '1' and 24 '0's.
 * The goal is to find the minimum number of moves (swapping adjacent rows or
 * columns) to move the '1' to the center of the matrix, which is at position (3, 3).
 *
 * Algorithm: Manhattan Distance
 * The minimum number of moves is the Manhattan distance between the '1's current
 * position (r, c) and the center (3, 3). The formula is |r - 3| + |c - 3|.
 *
 * 1. Find the coordinates of the '1': Iterate through the 5x5 matrix to find the
 *    1-indexed row `r` and column `c` where the '1' is located.
 * 2. Calculate the row moves: Determine the absolute difference between the
 *    current row `r` and the target row 3.
 * 3. Calculate the column moves: Determine the absolute difference between the
 *    current column `c` and the target column 3.
 * 4. The total number of moves is the sum of the row moves and column moves.
 */
#include <iostream>
#include <stdio.h>
#include <cmath> // For std::abs
using namespace std;
int main() {
  // freopen("in.txt","r",stdin);
  int i, j, in, r = 0, c = 0;
  
  /**
   * Block Logic: Find the 1-indexed coordinates of the '1' in the matrix.
   */
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> in;
      if (in == 1) {
        r = i; // Store the row (1-5)
        c = j; // Store the column (1-5)
        // A break here only exits the inner loop. The outer loop continues
        // unnecessarily but harmlessly.
        break;
      }
    }
  }
  
  // A more direct calculation would be: cout << abs(r - 3) + abs(c - 3);
  
  /**
   * Block Logic: Calculate the number of row moves required.
   * This is equivalent to calculating |r - 3|.
   */
  if (r == 5 || r == 1)
    in = 2; // 2 moves to get to row 3
  else if (r == 4 || r == 2)
    in = 1; // 1 move to get to row 3
  else
    in = 0; // 0 moves, already in row 3
    
  /**
   * Block Logic: Calculate the number of column moves required.
   * This is equivalent to calculating |c - 3|.
   * Note: The variable 'j' is confusingly reused here to store the result.
   */
  if (c == 5 || c == 1)
    j = 2; // 2 moves to get to column 3
  else if (c == 4 || c == 2)
    j = 1; // 1 move to get to column 3
  else
    j = 0; // 0 moves, already in column 3
    
  // The total number of moves is the sum of row moves and column moves.
  cout << in + j;
  
  return 0;
}
