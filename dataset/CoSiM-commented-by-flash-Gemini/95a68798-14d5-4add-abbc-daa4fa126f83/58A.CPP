/**
 * @file 58A.CPP
 * @brief Determines if the word "hello" can be formed as a subsequence from a given input string.
 * * Algorithm: Greedy Subsequence Matching.
 * Time Complexity: $O(N)$ where N is the length of the input string, due to a single pass.
 * Space Complexity: $O(1)$ for storing the target subsequence (string `s`).
 *
 * This program checks if the characters 'h', 'e', 'l', 'l', 'o' appear in the correct
 * order within the input string, not necessarily contiguously.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams and unties
   * cin and cout to optimize high-throughput I/O required for competitive programming constraints.
   */
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  string str; // The input string to be checked.
  cin >> str;

  string s; // The string being built to match "hello".
  /**
   * Block Logic: Iterates through the input string `str` to greedily find the characters
   * of "hello" in sequential order.
   * Pre-condition: `str` contains the input string. `s` is empty.
   * Invariant: At each step, `s` contains the longest prefix of "hello" found as a subsequence so far.
   * Post-condition: `s` will contain "hello" if it can be formed as a subsequence, otherwise a partial string.
   */
  for (int i = 0; i < str.size(); i++) {
    // Inline: If `s` is empty and the current character is 'h', start forming "hello".
    if (s.empty() && str[i] == 'h')
      s.push_back('h');
    // Inline: If `s` is "h" and the current character is 'e', append 'e'.
    else if (s == "h" && str[i] == 'e')
      s.push_back('e');
    // Inline: If `s` is "he" and the current character is 'l', append 'l'.
    else if (s == "he" && str[i] == 'l')
      s.push_back('l');
    // Inline: If `s` is "hel" and the current character is 'l', append the second 'l'.
    else if (s == "hel" && str[i] == 'l')
      s.push_back('l');
    // Inline: If `s` is "hell" and the current character is 'o', append 'o' to complete "hello".
    else if (s == "hell" && str[i] == 'o')
      s.push_back('o');
  }

  // Block Logic: Checks if the constructed string `s` is exactly "hello".
  // Post-condition: Prints "YES" if "hello" was found as a subsequence, "NO" otherwise.
  (s == ("hello")) ? cout << "YES\n" : cout << "NO\n";
  return 0; // Added explicit return 0 for main, though often implicitly handled.
}