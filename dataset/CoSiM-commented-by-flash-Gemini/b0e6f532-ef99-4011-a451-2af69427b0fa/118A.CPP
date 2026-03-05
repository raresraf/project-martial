/**
 * @file 118A.CPP
 * @brief Solution to a string processing problem where the task is to transform
 *        an input string by removing all vowels and inserting a "." before
 *        each consonant, converting the consonant to lowercase.
 *
 * Algorithm: Linear scan with character classification and transformation.
 *            The algorithm iterates through the input string character by character.
 *            For each character, it checks if it's a vowel (case-insensitive).
 *            If it's a consonant, a '.' is printed, and the consonant is printed
 *            in lowercase. Vowels are skipped.
 * Time Complexity: O(N * K), where N is the length of the input string and K is the
 *                  length of the vowel set. In practice, K is a small constant,
 *                  so it's effectively O(N).
 * Space Complexity: O(1), excluding input and output string storage.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  // Functional Utility: Defines a string containing all vowels (both lowercase and uppercase)
  // for efficient lookup.
  string f = "aoyeuiAOYEUI";
  // Block Logic: Reads the input string to be processed.
  cin >> s;
  // Functional Utility: Initializes loop counter `i` to -1, which is pre-incremented
  // before each iteration, effectively starting from index 0.
  int i = -1;
  /**
   * Block Logic: Iterates through each character of the input string to apply
   *              the transformation rules.
   * Invariant: At each iteration, `s[i]` is the current character being processed.
   *            Characters are processed from left to right.
   */
  while (++i < s.size()) {
    // Pre-condition: Checks if the current character `s[i]` is a vowel.
    // Functional Utility: `f.find(s[i]) != string::npos` efficiently checks if `s[i]`
    // exists within the predefined vowel string `f`.
    if (f.find(s[i]) != string::npos)
      // Block Logic: If the character is a vowel, it is skipped.
      continue;
    else
      // Block Logic: If the character is a consonant, prints a '.' followed by
      // the lowercase version of the consonant.
      // Functional Utility: `s[i] > 'Z' ? s[i] : s[i] + ('a' - 'A')` is a ternary
      // operator that converts an uppercase character to lowercase, otherwise
      // keeps it as is.
      cout << "." << (char)(s[i] > 'Z' ? s[i] : s[i] + ('a' - 'A'));
  }
  // Post-condition: The transformed string is printed to standard output implicitly
  // by the `cout` operations within the loop.
  // The function implicitly returns 0 as it reaches the end.
  return 0;
}
