/**
 * @file 58A.CPP
 * @brief Solution to Codeforces Problem 58A: "Chat room".
 *        The problem asks to determine if the word "hello" can be formed as a subsequence
 *        from a given input string. A subsequence is formed by deleting zero or more
 *        characters from a string without changing the order of the remaining characters.
 *
 * Algorithm: Two-pointer approach. One pointer iterates through the input string,
 *            and another pointer tracks the characters of "hello" that have been found so far.
 *            If the current character in the input string matches the character pointed to
 *            in "hello", the "hello" pointer advances.
 * Time Complexity: O(L), where L is the length of the input string, as it involves a single
 *                  pass through the input string.
 * Space Complexity: O(1) beyond input storage, as only a few variables are used.
 */
#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <string>

using namespace std;

int main() {
  /**
   * Functional Utility: Reads the input string from which to form the subsequence.
   * `line`: Stores the input string.
   * `temp`: Stores the target word "hello" to be found as a subsequence.
   * `index`: Tracks the progress of matching characters in the `temp` string.
   */
  string line;
  cin >> line;
  string temp = "hello";
  int index = 0;
  /**
   * Block Logic: Iterates through the input string `line` and attempts to find "hello" as a subsequence.
   * Pre-condition: `line` is the input string, `temp` is "hello", `index` starts at 0.
   * Invariant: `index` represents the number of characters of "hello" found consecutively as a subsequence.
   */
  for (auto i = 0; i < line.length() && index < temp.length(); i++) {
    /**
     * Conditional Logic: Checks if the current character in `line` matches the next expected character in `temp`.
     * If a match is found, increment `index` to look for the next character of "hello".
     */
    if (line[i] == temp[index]) {
      index++;
    }
  }
  /**
   * Conditional Logic: Determines if the entire word "hello" was found as a subsequence.
   * If `index` equals the length of "hello", it means all characters were found in order.
   */
  if (index == temp.length())
    cout << "YES"; // Output "YES" if "hello" is a subsequence.
  else
    cout << "NO";  // Output "NO" if "hello" is not a subsequence.
  return 0;
}