/**
 * @file 282A.CPP
 * @brief Solution for the "Bit++" problem from Codeforces using string search.
 * @details This program calculates the final value of a variable 'X' after a series
 * of increment ("++X", "X++") and decrement ("--X", "X--") operations.
 *
 * @algorithm String Find
 * This approach robustly identifies the type of operation by searching for the
 * existence of a '+' character within the operation string.
 * 1. Read the number of operations, `n`.
 * 2. Initialize a counter `c` to 0.
 * 3. Loop `n` times.
 * 4. In each iteration, read the operation string.
 * 5. Use `std::string::find` to check if a '+' character is present.
 * 6. If '+' is found, increment the counter.
 * 7. Otherwise (if no '+' is found), the operation must be a decrement, so decrement the counter.
 * 8. After the loop, print the final value of the counter.
 *
 * @complexity
 * - Time: O(N), where N is the number of statements. The `find` operation on a
 *   3-character string is effectively constant time.
 * - Space: O(1), as it only stores the current operation string and a few variables.
 */

#include <iostream>
#include <string>

using namespace std;

int main() {
  // n: the number of statements to execute.
  // c: the counter, representing the value of X, initialized to 0.
  int n, c = 0;
  cin >> n;

  /**
   * @brief Process each of the `n` statements.
   */
  while (n--) {
    // s: stores the current operation string (e.g., "++X").
    string s;
    cin >> s;

    /**
     * @brief Check if the operation is an increment or decrement.
     * `s.find('+')` searches for the '+' character in the string.
     * It returns `string::npos` if the character is not found.
     * This is a robust way to detect an increment operation.
     */
    // Pre-condition: Check if the string contains a '+'.
    if (s.find('+') != string::npos) {
      c++; // Increment if '+' is found.
    } else {
      c--; // Decrement if no '+' is found (must be a decrement operation).
    }
  }

  // Print the final value of the counter.
  cout << c << endl;

  return 0; // Indicate successful execution.
}
