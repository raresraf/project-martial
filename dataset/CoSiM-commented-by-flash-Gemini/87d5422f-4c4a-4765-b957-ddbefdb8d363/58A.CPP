/**
 * @file 58A.CPP
 * @brief Solution to a problem that checks if the string "hello" is a subsequence
 *        of a given input string. A subsequence is formed by deleting zero
 *        or more characters from another string without changing the order
 *        of the remaining characters.
 *
 * Algorithm: Two-pointer approach for subsequence matching.
 *            The algorithm uses two pointers: one for iterating through the
 *            input string `s` and another for tracking the characters of
 *            the target subsequence "hello". It increments the "hello" pointer
 *            only when a matching character is found in `s`.
 * Time Complexity: O(N), where N is the length of the input string `s`.
 *                  The input string is traversed at most once.
 * Space Complexity: O(1), excluding input and output string storage.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output operations.
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  string s;
  // Block Logic: Reads the input string from standard input.
  cin >> s;
  // Functional Utility: Defines the target subsequence "hello".
  string hello = "hello";
  // Functional Utility: `cidx` acts as a pointer for the "hello" string,
  // tracking how many characters of "hello" have been found so far.
  int cidx = 0;
  /**
   * Block Logic: Iterates through the input string `s` to find "hello" as a subsequence.
   * Invariant: `cidx` represents the number of characters of "hello" found consecutively
   *            as a subsequence in `s` up to the current `i`.
   */
  for (int i = 0; i < (int)s.length() && cidx < 5; i++) {
    // Pre-condition: Checks if the current character in `s` matches the
    //               next expected character in "hello".
    if (s[i] == hello[cidx]) {
      // Functional Utility: If a match is found, advances the "hello" pointer.
      cidx++;
    }
  }
  // Post-condition: After iterating through `s`, checks if all characters of "hello"
  //                were found as a subsequence.
  if (cidx >= 5) {
    // Block Logic: If `cidx` is 5 (meaning all characters of "hello" were found),
    // prints "YES".
    cout << "YES" << endl;
  } else {
    // Block Logic: Otherwise, prints "NO".
    cout << "NO" << endl;
  }
  return 0;
}
