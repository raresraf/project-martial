/**
 * @file 263A.CPP
 * @brief A concise solution for the "Beautiful Matrix" competitive programming problem.
 * @details This program calculates the Manhattan distance required to move a single '1'
 * in a 5x5 matrix to the center position (3,3). This version is optimized to
 * compute and print the result as soon as the '1' is found, without storing the
 * entire matrix.
 *
 * @section Algorithm
 * 1. Iterate through a 5x5 grid using nested loops (from 0 to 4).
 * 2. In each iteration, read a single integer value.
 * 3. If the value is non-zero (i.e., it's the '1'), immediately calculate the
 *    Manhattan distance from the current 0-indexed coordinates (i, j) to the
 *    center (2, 2). The formula used is |2 - i| + |2 - j|.
 * 4. Print the result. The program will continue reading the remaining input,
 *    but only the first result printed corresponds to the '1'.
 *
 * @section Complexity
 * - Time Complexity: O(1) on average. In the worst case, it iterates through all 25
 *   cells, but since the matrix size is fixed, this is considered constant time.
 * - Space Complexity: O(1), as it only stores the most recently read integer, not
 *   the entire matrix.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Functional Utility: Decouples C++ streams from standard C streams
  // to optimize high-throughput I/O.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  
  int n = 5, v;
  /**
   * Block Logic: Iterate through the 5x5 grid, reading one element at a time.
   * Invariant: The loops scan the grid from (0,0) until all inputs are consumed.
   */
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      cin >> v;
      /**
       * Block Logic: Check if the current element is the target value '1'.
       * Pre-condition: 'v' holds the value of the cell at (i, j).
       * In this problem, the target '1' is the only non-zero value.
       */
      if (v)
        // Post-condition: If the target is found, calculate the Manhattan distance
        // from the current 0-indexed position (i,j) to the center (2,2) and print the result.
        cout << abs(2 - i) + abs(2 - j) << "
";
    }
}
