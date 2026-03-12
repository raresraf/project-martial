/**
 * @57ba41d7-78eb-443c-a786-cffd2177f931/112A.CPP
 * @brief This program performs a case-insensitive lexicographical comparison of two input strings.
 * It outputs -1 if the first string is lexicographically smaller, 1 if it is larger,
 * and 0 if both strings are equal.
 *
 * Algorithm: Character-by-character comparison after converting to lowercase.
 * Time Complexity: O(L) where L is the length of the strings (assuming equal length).
 * Space Complexity: O(1) (modifies strings in-place).
 */

// link : https://codeforces.com/contest/112/problem/A

#include <bits/stdc++.h> // A common header in competitive programming; includes many standard libraries.

using namespace std; // Uses the standard namespace to avoid prefixing std::.

string s1, s2; // Declares two global string variables to store the input strings.

int k; // Declared but not used in the provided snippet.

int main() { // Corrected main function signature to return int.

  cin >> s1 >> s2; // Reads the two input strings from standard input.

  /**
   * Block Logic: Iterates through the strings, comparing characters lexicographically after converting them to lowercase.
   * Invariant: At each step, characters up to 'i-1' have been found to be equal.
   */
  for (int i = 0; i < s1.size(); i++) { // Loops through the characters of the strings.
                                         // Assumes both strings have at least this length, or comparison is only needed up to s1.size().

    /**
     * Block Logic: Converts uppercase characters to lowercase for case-insensitive comparison.
     * Pre-condition: 's1[i]' and 's2[i]' are valid characters.
     * Invariant: 's1[i]' and 's2[i]' are in lowercase before comparison.
     */
    if (s1[i] < 'a') // Inline: Checks if s1[i] is an uppercase letter (ASCII 'A' through 'Z' are less than 'a').
      s1[i] += ' ';  // Inline: Adds ASCII 32 to convert an uppercase letter to its lowercase equivalent.

    if (s2[i] < 'a') // Inline: Checks if s2[i] is an uppercase letter.
      s2[i] += ' ';  // Inline: Adds ASCII 32 to convert an uppercase letter to its lowercase equivalent.

    /**
     * Block Logic: Compares the lowercase characters and determines the lexicographical order.
     * Invariant: If a difference is found, the program outputs the result and terminates immediately.
     */
    if (s1[i] < s2[i]) { // Inline: If s1[i] is lexicographically smaller than s2[i].
      cout << "-1";     // Output -1.
      return 0;         // Terminate the program.
    }
    if (s1[i] > s2[i]) { // Inline: If s1[i] is lexicographically greater than s2[i].
      cout << "1";      // Output 1.
      return 0;         // Terminate the program.
    }
  }

  // If the loop completes, it means all characters were equal, so the strings are identical.
  cout << "0"; // Output 0.
  return 0;    // Indicates successful program termination.
}
