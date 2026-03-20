/**
 * @file 282A.CPP
 * @brief A minimal and efficient solution for Codeforces problem 282A, "Bit++".
 * @details This program evaluates a series of increment and decrement operations
 * by using a concise check on the operation string.
 *
 * @section Algorithm
 * The solution leverages a key observation about the input format to simplify
 * the logic significantly.
 *
 * 1. Read the number of operations, `n`.
 * 2. Initialize a counter, `ans`, to 0.
 * 3. Loop `n` times, reading one operation string `s` in each iteration.
 * 4. The core of the algorithm is checking the character at index 1 of the string `s`.
 *    For all possible inputs ("++X", "X++", "--X", "X--"), this character is
 *    uniquely '-' for a decrement and '+' for an increment.
 * 5. If `s[1]` is '-', the counter `ans` is decremented. Otherwise, it is incremented.
 * 6. After the loop completes, the final value of `ans` is printed.
 *
 * @subsection Complexity
 * - Time Complexity: O(n), where `n` is the number of operations, due to the single loop.
 * - Space Complexity: O(1), as the memory for the string `s` is reused and does not
 *   depend on `n`.
 */
#include <iostream>

using namespace std;

int main() {
  // n: the number of operations to be performed.
  int n;
  cin >> n;
  string s;
  // ans: the variable that will be incremented or decremented, starts at 0.
  int ans = 0;
  /**
   * @brief This loop iterates through all `n` operations.
   */
  for (int i = 0; i < n; ++i) {
    cin >> s;
    /**
     * @brief This conditional statement efficiently determines the operation type.
     * It checks the character at index 1, which is sufficient to distinguish
     * between an increment ('+') and a decrement ('-').
     */
    if (s[1] == '-')
      --ans;
    else
      ++ans;
  }
  // Print the final value of the counter.
  cout << ans;
}