/**
 * @file 58A.CPP
 * @brief Solution for Codeforces Problem 58A - "Chat Room".
 * This program determines if the word "hello" is a subsequence of a given input string.
 * A subsequence is formed by deleting zero or more characters from another string
 * without changing the order of the remaining characters.
 *
 * @algorithm
 * The algorithm iterates through the input string `s` to find the characters
 * 'h', 'e', 'l', 'l', 'o' in sequential order. It uses an index `j` to track
 * the current character of "hello" being searched for. When a match is found,
 * it "marks" the matched character in the "hello" string `t` (by setting it to 0)
 * and advances `j`. After iterating through the entire input string, it checks
 * if all characters of "hello" were successfully matched by verifying the last
 * character of `t`.
 *
 * Time Complexity: O(N), where N is the length of the input string `s`.
 *   The program performs a single pass through the input string.
 * Space Complexity: O(1), as only a few fixed-size variables and strings are used.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @brief Main function to determine if "hello" is a subsequence of the input string.
   * Reads an input string `s` and checks if the characters of "hello" appear
   * in the correct order as a subsequence.
   * @return 0 on successful execution.
   */
  string t = "hello"; ///< @var t The target subsequence string, initialized to "hello". Characters are marked as found by setting them to 0.
  string s; ///< @var s The input string from which to find the subsequence.
  cin >> s; ///< Read the input string.

  int j = 0; ///< @var j Index to track the current character of `t` (the "hello" string) being searched for.

  /**
   * @brief Iterates through the input string `s` to find characters of `t` ("hello") sequentially.
   * @invariant `j` points to the next character in `t` that needs to be found.
   * `s[i]` is the current character being examined in the input string.
   * @post If "hello" is a subsequence, `j` will have advanced past the last character of `t`,
   * and the corresponding character in `t` will be marked as 0.
   */
  for (int i = 0; s[i] != '\0'; i++) {
    /**
     * @brief Checks if the current character in `s` matches the next required character in `t`.
     * @details If a match is found, the character in `t` is "consumed" (marked as 0),
     * and the index `j` advances to look for the next character of "hello".
     */
    if (s[i] == t[j]) {
      t[j++] = 0; // Mark the character in `t` as found and advance `j`.
    }
  }

  /**
   * @brief Determines and prints the result based on whether "hello" was found as a subsequence.
   * @details Checks if the last character of the `t` string (which was 'o' initially) has been
   * marked as 0. If it has, it means all characters of "hello" were found in order.
   */
  if (t[t.length() - 1] == 0)
    cout << "YES\n"; // All characters of "hello" found as a subsequence.
  else
    cout << "NO\n"; // Not all characters of "hello" found in order.
  return 0;
}
