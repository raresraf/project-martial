/**
 * @file 282A.CPP
 * @brief Solution to a competitive programming problem that simulates a simple counter.
 * The program initializes a counter to zero, then processes a given number of operations.
 * Each operation is represented by a string. If the string contains a '+' character,
 * the counter is incremented; otherwise, it is decremented. The final value of the
 * counter is printed.
 * Algorithm: Iterative counter manipulation based on string content.
 * Time Complexity: O(N * L), where N is the number of operations and L is the average
 * length of the operation strings (due to string search).
 * Space Complexity: O(L) for storing the input string.
 */
#include <algorithm> // Not used in this specific implementation, common in competitive programming.
#include <cmath>     // Not used in this specific implementation, common in competitive programming.
#include <iostream>
#include <stdio.h>   // For printf/scanf which are not used directly, but part of bits/stdc++.h concept.
#include <string>
#include <vector>    // Not used in this specific implementation, common in competitive programming.
using namespace std;

// Type definition for vector of integers, common in competitive programming for brevity.
typedef vector<int> vi;
// Macro to get iterators for the beginning and end of a container, used for algorithms.
#define all(c) (c).begin(), (c).end()
// Macro for push_back, a common operation on vectors.
#define pb push_back
// Macro for tracing/debugging vectors, not used in the final logic but for development.
#define tr(a)                                                                  \
  cout << "Vector: ";                                                          \
  for (auto i : a)                                                             \
    cout << i << " ";                                                          \
  cout << "\n";

int main() {
  /**
   * Functional Utility: Optimizes C++ standard stream I/O operations.
   * `ios_base::sync_with_stdio(false)` unties C++ streams from C standard streams,
   * which can be slow. `cin.tie(NULL)` prevents `cin` from flushing `cout`
   * before each input operation, further speeding up I/O. These are common
   * competitive programming optimizations for performance-critical tasks.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  
  // Declares 'n' for the number of operations, 'i' as a loop counter, and 'c' as the main counter.
  int n, i, c = 0;
  // Reads the total number of operations 'n'.
  cin >> n;
  // Declares a string 's' to store each operation.
  string s;
  
  /**
   * Block Logic: Iterates 'n' times to process each operation.
   * Invariant: At the start of each iteration, 'i' is the current operation index (0-based).
   * Post-condition: 'c' reflects the cumulative result of all operations.
   */
  for (i = 0; i < n; i++) {
    // Reads the operation string.
    cin >> s;
    
    /**
     * Conditional Logic: Modifies the counter 'c' based on the presence of '+' in the string.
     * Uses the ternary operator for concise conditional increment/decrement.
     * `s.find('+') != s.npos` checks if the '+' character exists in the string.
     * If true, `c` is incremented by 1; otherwise, `c` is decremented by 1.
     */
    c += (s.find('+') != s.npos) ? 1 : -1;
  }
  // Prints the final value of the counter 'c'.
  cout << c;
  
  return 0; // Indicates successful program execution.
}
