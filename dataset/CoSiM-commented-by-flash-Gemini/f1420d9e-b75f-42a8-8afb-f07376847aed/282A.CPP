/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces (Problem 282A).
 *        This program simulates operations on a single variable 'ans'.
 *        It reads 't' operation strings. Each string is parsed by checking
 *        characters at specific indices (0 and 2) to determine if 'ans'
 *        should be incremented or decremented.
 *        Note: This implementation may increment/decrement 'ans' twice for
 *        operations like "X++" or "X--" due to separate checks for `s[0]` and `s[2]`.
 * Algorithm: Counter Simulation / String Character Checking
 * Time Complexity: O(T * C), where T is the number of operations and C is a small
 *                  constant number of character checks per string (effectively O(T)).
 * Space Complexity: O(L) to store the operation string (L is fixed at 3), effectively O(1).
 */

#include <bits/stdc++.h> // Includes various standard libraries, e.g., iostream and string.
using namespace std;

/**
 * @brief Main function to process Bit++ operations and determine the final value of 'ans'.
 * @return 0 if the program executes successfully.
 */
int main() {
  int t; // Declare an integer to store the number of operations.
  // Read the number of operations.
  cin >> t;
  int ans = 0; // Initialize the counter variable 'ans' to 0.

  /**
   * Block Logic: Processes 't' operation strings, modifying 'ans' based on character checks.
   * Invariant: 'ans' reflects the cumulative result of operations processed so far.
   */
  for (int i = 0; i < t; i++) {
    // int ans=0; // Commented out: This line would incorrectly reset 'ans' in each iteration.
    string s; // Declare a string to store the current operation.
    // Read the operation string.
    cin >> s;
    
    /**
     * Block Logic: Checks the character at index 0 of the string.
     * Observed Behavior: If `s[0]` is '+', `ans` is incremented. If `s[0]` is '-', `ans` is decremented.
     *                    This handles "++X" and "--X" correctly for `s[0]`.
     */
    if (s[0] == '+') {
      ++ans; // Increment 'ans'.
    }
    if (s[0] == '-') {
      --ans; // Decrement 'ans'.
    }
    
    /**
     * Block Logic: Checks the character at index 2 of the string.
     * Observed Behavior: If `s[2]` is '+', `ans` is incremented. If `s[2]` is '-', `ans` is decremented.
     *                    This handles "X++" and "X--" for `s[2]`.
     *                    For "X++" and "X--", `ans` would also be affected by the checks for `s[0]`,
     *                    resulting in a double increment or decrement.
     */
    if (s[2] == '+') {
      ans++; // Increment 'ans'.
    }
    if (s[2] == '-') {
      ans--; // Decrement 'ans'.
    }
    // cout << ans << endl; // Commented out: This line would print 'ans' after each operation.
  }
  // Print the final value of 'ans' followed by a newline.
  cout << ans << endl;

  return 0; // Indicate successful program termination.
}
