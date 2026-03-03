/**
 * @file 263A.CPP
 * @brief Solution for Codeforces Problem 263A - "Beautiful Matrix".
 * This program calculates the minimum number of moves required to bring the
 * '1' in a 5x5 matrix to the center position (2,2, 0-indexed).
 * A move consists of swapping adjacent rows or columns. This is equivalent
 * to finding the Manhattan distance between the '1' and the center.
 * Algorithm: Iterative search for '1' and Manhattan distance calculation.
 * Time Complexity: O(1) - The matrix size is fixed (5x5), so the number of operations is constant.
 * Space Complexity: O(1) - Uses a constant amount of memory for variables.
 */
#include <algorithm> // Provides functions like std::abs
#include <cctype>    // Character manipulation functions
#include <climits>   // Limits of integral types
#include <cmath>     // Mathematical functions like abs for floating-point types (though integer abs is in cstdlib)
#include <complex>   // Complex number type
#include <cstdio>    // C-style input/output functions
#include <cstdlib>   // General utilities, including abs for integers
#include <functional> // Function objects
#include <iomanip>    // Output formatting
#include <iostream>   // Standard input/output streams
#include <iterator>   // Iterator primitives
#include <map>        // Map container
#include <set>        // Set container
#include <sstream>    // String stream
#include <string>     // String type
#include <utility>    // Pairs and other utility classes
#include <vector>     // Vector container

// Macro for a common loop pattern: for (int i = 0; i < n; ++i)
#define REP(i, n) for (int i = 0; i < (int)(n); ++i)
// Macro for a common loop pattern: for (int i = m; i < n; ++i)
#define FOR(i, m, n) for (int i = (m); i < (int)(n); ++i)
// Macro to get iterators for the entire range of a container
#define ALL(x) (x).begin(), (x).end()
// Defines a large integer value to represent infinity
#define INF 2000000000

// Conditional compilation for debugging statements in a local environment.
#ifdef LOCAL
#define dprintf(...) fprintf(stdout, __VA_ARGS__) // Prints debug messages to stdout if LOCAL is defined.
#else
#define dprintf(...) 0 // Otherwise, debug messages are compiled out.
#endif

using namespace std; // Uses the standard namespace to avoid prefixing std::

typedef long long LL;       // Alias for long long
typedef unsigned long long ULL; // Alias for unsigned long long
typedef unsigned int uint; // Alias for unsigned int

// Defines the mathematical constant PI.
const double PI = 3.1415926535897932384626433832795;

int main() {
  char c;        // Variable to store the character read from input (either '0' or '1').
  int answer = 0; // Variable to store the calculated minimum moves.
  bool found = false; // Flag to indicate if the '1' has been found in the matrix.
  
  /**
   * Block Logic: Iterates through the 5x5 matrix to find the position of '1'.
   * The loops break early once '1' is found to optimize performance.
   * Invariant: 'answer' stores the Manhattan distance to the center from the position of '1'.
   */
  REP(i, 5) { // Loop through rows (0-indexed from 0 to 4).
    REP(j, 5) { // Loop through columns (0-indexed from 0 to 4).
      cin >> c; // Reads the character for the current cell.
      // Checks if the current character is '1'.
      if (c == '1') {
        // Calculates the Manhattan distance: |row_of_1 - center_row| + |col_of_1 - center_col|.
        // The center of a 5x5 matrix (0-indexed) is at (2, 2).
        answer = abs(2 - i) + abs(2 - j);
        found = true; // Sets the flag to true as '1' has been found.
      }
      if (found) // If '1' was found in the current row, no need to check further columns in this row.
        break;
    }
    if (found) // If '1' was found in any row, no need to check further rows.
      break;
  }
  
  // Prints the calculated minimum number of moves to bring '1' to the center.
  cout << answer << endl;

  return 0; // Indicates successful program execution.
}

