/**
 * @file 112A.CPP
 * @brief Solution for the "Petya and Strings" problem from Codeforces.
 *
 * This problem requires comparing two strings lexicographically, ignoring case.
 * The comparison should output:
 * - 0 if the strings are equal.
 * - 1 if the first string is lexicographically greater than the second.
 * - -1 if the first string is lexicographically smaller than the second.
 *
 * Problem Link: https://codeforces.com/problemset/problem/112/A
 *
 * Algorithm: Case-insensitive character-by-character comparison.
 * Time Complexity: O(L), where L is the length of the strings.
 * Space Complexity: O(L) for storing the strings.
 */

#include <bits/stdc++.h> // Includes common C++ standard library headers.
using namespace std;

int main() {
  string s1, s2; // Declares two strings to store the input.
  // Reads the first string from standard input.
  getline(cin, s1);
  // Reads the second string from standard input.
  getline(cin, s2);
  
  // Converts both strings to lowercase for case-insensitive comparison.
  // Functional Utility: Ensures that 'A' and 'a' are treated as the same character.
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
  
  int res = 0; // Stores the comparison result: 0 (equal), 1 (s1 > s2), -1 (s1 < s2).
               // Initialized to 0, assuming equality until a difference is found.

  /**
   * Block Logic: Performs a character-by-character lexicographical comparison.
   * Invariant: `res` holds the comparison status based on the *last* difference encountered.
   *            For a standard lexicographical comparison, the loop would typically
   *            terminate upon finding the first differing character.
   */
  for (int i = 0; i < s1.length(); i++) {
    // If a character in `s1` is greater than the corresponding character in `s2`.
    if (s1[i] > s2[i]) {
      res = 1; // Mark `s1` as greater.
      // @note For strict lexicographical comparison, a `break` would typically be here.
      //       The current implementation continues the loop.
    }
    // If a character in `s1` is less than the corresponding character in `s2`.
    else if (s1[i] < s2[i]) {
      res = -1; // Mark `s1` as smaller.
      // @note For strict lexicographical comparison, a `break` would typically be here.
      //       The current implementation continues the loop.
    }
    // If characters are equal, `res` remains 0 (or retains a previous non-zero value).
  }
  
  // Prints the final comparison result.
  cout << res;
  return 0;
}
