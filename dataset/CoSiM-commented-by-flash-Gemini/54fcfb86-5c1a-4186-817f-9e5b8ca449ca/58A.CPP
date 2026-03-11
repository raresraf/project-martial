/**
 * @file 58A.CPP
 * @brief Solution to Codeforces Problem 58A: "Chat room".
 *        The problem asks to determine if the word "hello" can be formed as a subsequence
 *        from a given input string. A subsequence is formed by deleting zero or more
 *        characters from a string without changing the order of the remaining characters.
 *
 * Algorithm: Two-pointer approach. One pointer (`b`) iterates through the input string,
 *            and another pointer (`a`) tracks the characters of "hello" that have been found so far.
 *            If the current character in the input string matches the character pointed to
 *            in "hello", the "hello" pointer advances.
 * Time Complexity: O(L), where L is the length of the input string, as it involves a single
 *                  pass through the input string.
 * Space Complexity: O(1) beyond input storage, as only a few variables are used.
 */
#include <iostream>
using namespace std;

int main() {
  /**
   * Functional Utility: Declaration of variables for string processing.
   * `greeting`: The target word "hello" to be found as a subsequence.
   * `line`: Stores the input string from which to form the subsequence.
   * `a`: Pointer/index for the `greeting` string, tracking characters found.
   * `b`: Pointer/index for the `line` string, iterating through its characters.
   * `output`: Stores the final result, "YES" or "NO".
   */
  const string greeting = "hello";
  string line;
  getline(cin, line); // Read the entire input line, including spaces.
  int a = 0, b = 0;
  string output = "NO";
  /**
   * Block Logic: Iterates through the input string `line` to find "hello" as a subsequence.
   * `b` advances through `line`, `a` advances through `greeting` only upon a match.
   * Pre-condition: `line` contains the input string, `greeting` is "hello", `a` and `b` start at 0.
   * Invariant: `a` represents the number of characters of "hello" found consecutively so far.
   */
  while (b < line.size()) {
    /**
     * Conditional Logic: Checks if the current character in `line` matches the next expected character in `greeting`.
     * If a match is found, increment `a` to look for the next character of "hello".
     * `b` is incremented in every iteration to move to the next character in `line`.
     */
    if (line[b++] == greeting[a]) {
      a++;
    }
    /**
     * Conditional Logic: Checks if the entire "hello" word has been found as a subsequence.
     * If `a` reaches the length of `greeting`, it means all characters were found in order.
     * Set `output` to "YES" and break the loop as the condition is met.
     */
    if (a >= greeting.size()) {
      output = "YES";
      break;
    }
  }
  // Output: Print the result ("YES" or "NO") followed by a newline.
  cout << output << endl;
  return 0;
}
