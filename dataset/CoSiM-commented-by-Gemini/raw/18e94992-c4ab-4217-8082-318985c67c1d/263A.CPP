/**
 * @file 263A.CPP
 * @brief Solution to the "Beautiful Matrix" competitive programming problem.
 * @details The problem involves a 5x5 matrix containing 24 zeros and a single digit '1'.
 * The goal is to find the minimum number of moves (swapping adjacent rows or columns)
 * required to move the '1' to the center of the matrix, which is at position (3,3)
 * in 1-based indexing or (2,2) in 0-based indexing.
 *
 * @section algorithm Algorithm
 * The minimum number of moves required is equivalent to the Manhattan distance between the
 * current position of the '1' and the center of the matrix. The Manhattan distance
 * between two points (x1, y1) and (x2, y2) is `|x1 - x2| + |y1 - y2|`.
 *
 * 1. The program iterates through the 5x5 matrix using nested loops with 0-based
 *    row index `i` and column index `j`.
 * 2. It reads the matrix elements one by one without storing the full matrix.
 * 3. When an element with value '1' is found at position (`i`, `j`), it calculates the
 *    Manhattan distance to the center (2, 2): `abs(i - 2) + abs(j - 2)`.
 * 4. This distance represents the minimum number of moves. The program prints this
 *    value and immediately terminates, as there is only one '1' in the matrix.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(1), as the loops run at most 25 times. On average, it's even
 *   faster since the program terminates as soon as the '1' is found.
 * - Space Complexity: O(1), as the program does not store the matrix, processing
 *   elements as they are read.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  // This preprocessor block is a common competitive programming practice.
  // It redirects standard input from "input" and standard output to "output"
  // when the code is compiled and run locally (i.e., not on an online judge).
#ifndef ONLINE_JUDGE
  freopen("input", "r", stdin);
  freopen("output", "w", stdout);
#endif

  int a; // Variable to store the value of the current matrix element.
  
  // Iterate through the 5x5 matrix to find the position of the '1'.
  for (int i = 0; i < 5; i++) {     // i represents the row index (0-4).
    for (int j = 0; j < 5; j++) { // j represents the column index (0-4).
      cin >> a;
      // If the current element is '1', its location is (i, j).
      if (a == 1) {
        // Calculate the Manhattan distance to the center (2, 2).
        // abs(i - 2) is the number of row swaps (up/down moves).
        // abs(j - 2) is the number of column swaps (left/right moves).
        cout << abs(i - 2) + abs(j - 2) << endl;
        // Terminate the program since the unique '1' has been found.
        return 0;
      }
    }
  }
}
