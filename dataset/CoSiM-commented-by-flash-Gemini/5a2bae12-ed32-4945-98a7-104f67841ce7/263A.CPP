/**
 * @5a2bae12-ed32-4945-98a7-104f67841ce7/263A.CPP
 * @brief This program solves problem 263A, commonly known as "Beautiful Matrix".
 * It reads a fixed 5x5 matrix where exactly one cell contains the value 1.
 * The objective is to determine the minimum number of moves required to bring
 * the cell containing 1 to the center of the matrix (which is at (2,2) for 0-indexed).
 * A move consists of swapping adjacent rows or adjacent columns.
 *
 * Algorithm: Manhattan distance calculation from the '1' to the center (2,2).
 * Time Complexity: O(1) as the matrix size (5x5) is constant.
 * Space Complexity: O(1) for storing the fixed-size matrix and a few variables.
 */

#include <bits/stdc++.h> // Includes many standard C++ library headers.
#include <string.h>      // Included but not directly used in this snippet.
using namespace std;     // Uses the standard namespace.

// Macro for fast I/O, common in competitive programming.
#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define endl "\n"                       // Defines endl as a newline character for faster output.
#define Max(x, y, z) max(x, max(y, z))  // Macro for finding maximum of three values.
#define Min(x, y, z) min(x, min(y, z))  // Macro for finding minimum of three values.
#define int long long // Redefines 'int' to 'long long' to prevent overflow in competitive programming.
#define trace1(x) cerr << #x << ": " << x << endl            // Debugging macro to print a single variable.
#define trace2(x, y) cerr << #x << ": " << x << " | " << #y << ": " << y << endl // Debugging macro for two variables.
#define trace3(x, y, z)                                                        \
  cerr << #x << ":" << x << " | " << #y << ": " << y << " | " << #z << ": "    \
       << z << endl // Debugging macro for three variables.

int32_t main() { // main function, using int32_t due to redefined 'int'.
  IOS; // Applies fast I/O optimizations.
  // Declares a 5x5 integer array (matrix) and two integers to store the row and column of '1'.
  int a[5][5], loc1, loc2; 
  
  /**
   * Block Logic: Reads the 5x5 matrix and finds the coordinates of the cell containing the value 1.
   * Invariant: After this nested loop, 'loc1' and 'loc2' store the 0-indexed row and column
   * of the cell containing the value 1.
   */
  for (int i = 0; i < 5; ++i) { // Outer loop iterates through rows (0 to 4).
    for (int j = 0; j < 5; ++j) { // Inner loop iterates through columns (0 to 4).
      cin >> a[i][j]; // Reads the integer value for the current cell.
      // Inline: Checks if the current cell contains the value 1.
      if (a[i][j] == 1) { 
        loc1 = i; // Stores the row index of '1'.
        loc2 = j; // Stores the column index of '1'.
      }
    }
  }
  
  // Block Logic: Calculates and prints the minimum number of moves.
  // The minimum moves correspond to the Manhattan distance from (loc1, loc2) to the center (2,2).
  // The absolute difference in row indices plus the absolute difference in column indices gives this distance.
  cout << (abs(loc2 - 2) + abs(loc1 - 2)); // Prints the calculated Manhattan distance.
  return 0; // Indicates successful program termination.
}